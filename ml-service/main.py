"""
FastAPI inference microservice.

Loads the fine-tuned DistilBERT backbone (safetensors) plus the two heads
(heads.safetensors) exported by ml/train.py and exposes:

    GET  /health   -> liveness probe
    POST /predict  -> body: { "text": "..." }
                       resp: { "emotion": {...}, "topics": [...] }

Environment:
    MODEL_DIR         (default: ./model)
    TOPIC_THRESHOLD   (default: 0.5)
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml-service")

MODEL_DIR       = os.environ.get("MODEL_DIR", "./model")
TOPIC_THRESHOLD = float(os.environ.get("TOPIC_THRESHOLD", "0.5"))
MAX_LEN         = 128


# ---------------------------------------------------------------------------
# Model wrapper (mirrors ml/train.py)
# ---------------------------------------------------------------------------
class MultiHeadTransformer(nn.Module):
    def __init__(self, backbone_dir: str, n_emotions: int, n_topics: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_dir)
        hidden = self.backbone.config.hidden_size
        self.emotion_head = nn.Linear(hidden, n_emotions)
        self.topic_head   = nn.Linear(hidden, n_topics)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.emotion_head(cls), self.topic_head(cls)


# ---------------------------------------------------------------------------
# Load once at startup
# ---------------------------------------------------------------------------
STATE: Dict = {}


def load_model():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f"MODEL_DIR '{MODEL_DIR}' not found. "
            f"Run `python ml/train.py` first."
        )
    with open(os.path.join(MODEL_DIR, "labels.json")) as f:
        meta = json.load(f)

    emotion_labels: List[str] = meta["emotion_labels"]
    topic_labels:   List[str] = meta["topic_labels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = MultiHeadTransformer(MODEL_DIR,
                                 n_emotions=len(emotion_labels),
                                 n_topics=len(topic_labels))
    heads = load_file(os.path.join(MODEL_DIR, "heads.safetensors"))
    model.emotion_head.weight.data.copy_(heads["emotion_head.weight"])
    model.emotion_head.bias.data.copy_(heads["emotion_head.bias"])
    model.topic_head.weight.data.copy_(heads["topic_head.weight"])
    model.topic_head.bias.data.copy_(heads["topic_head.bias"])
    model.to(device).eval()

    STATE.update({
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "emotion_labels": emotion_labels,
        "topic_labels":   topic_labels,
    })
    log.info(f"Loaded model from {MODEL_DIR} on {device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    STATE.clear()


app = FastAPI(title="Review Analyzer ML Service",
              version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class TopicScore(BaseModel):
    label: str
    score: float


class EmotionResult(BaseModel):
    label: str
    score: float
    distribution: Dict[str, float]


class PredictResponse(BaseModel):
    emotion: EmotionResult
    topics: List[TopicScore]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@torch.inference_mode()
def _infer(text: str) -> PredictResponse:
    tok = STATE["tokenizer"](
        text, truncation=True, padding="max_length",
        max_length=MAX_LEN, return_tensors="pt",
    ).to(STATE["device"])
    emo_logits, top_logits = STATE["model"](tok["input_ids"], tok["attention_mask"])

    emo_probs = torch.softmax(emo_logits, dim=-1)[0].cpu().tolist()
    top_probs = torch.sigmoid(top_logits)[0].cpu().tolist()

    emo_labels = STATE["emotion_labels"]
    top_labels = STATE["topic_labels"]

    top_idx = int(max(range(len(emo_probs)), key=lambda i: emo_probs[i]))
    emotion = EmotionResult(
        label=emo_labels[top_idx],
        score=round(float(emo_probs[top_idx]), 4),
        distribution={l: round(float(p), 4)
                      for l, p in zip(emo_labels, emo_probs)},
    )

    topics = sorted(
        [TopicScore(label=l, score=round(float(p), 4))
         for l, p in zip(top_labels, top_probs)
         if p >= TOPIC_THRESHOLD],
        key=lambda t: t.score, reverse=True,
    )
    # If nothing crosses threshold, still surface top-1 for UX
    if not topics:
        i = int(max(range(len(top_probs)), key=lambda i: top_probs[i]))
        topics = [TopicScore(label=top_labels[i],
                             score=round(float(top_probs[i]), 4))]

    return PredictResponse(emotion=emotion, topics=topics)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return _infer(req.text.strip())
    except Exception as e:  # noqa: BLE001
        log.exception("inference failure")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
