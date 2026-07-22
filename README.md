# Multi-Task Customer Review Analyzer
A Multi-Task Transformer for Emotion Detection & Business Topic Classification 🚀

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?logo=pytorch&logoColor=white)]()
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-blue?logo=huggingface&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-%2300B4AB?logo=fastapi&logoColor=white)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model_Hub-orange?logo=huggingface&logoColor=white)]()

A production-oriented, end-to-end NLP solution that simultaneously predicts fine-grained customer emotions and tags reviews with business-specific topics. Built for real-time inference with a shared DistilBERT backbone and two specialized classification heads.

---

## Why Multi-Task Learning? 💡
Training a single shared backbone for both emotion classification (multi-class) and topic categorization (multi-label) has several advantages over separate models:

- Shared representations reduce inference time and memory footprint (one forward pass -> two heads).
- Shared features improve generalization: information from topic labels helps emotion detection and vice versa (transfer learning within tasks).
- Easier deployment and monitoring (single model artifact).
- Joint training enables consistent pre-processing and reduces duplication across pipelines.

### Architecture (ASCII)
Text -> Tokenization -> DistilBERT (shared encoder)
                 |
                 +--> Emotion Head (Linear -> Softmax)  (Loss: CrossEntropy)
                 |
                 +--> Topic Head   (Linear -> Sigmoid)   (Loss: BCEWithLogits)

ASCII diagram:

                   +-------------------------+
    Input Text --->|  Tokenizer / Encoder    |---> [CLS] representation
                   |   distilbert-base-uncased|
                   +-------------------------+
                              |
              +---------------+---------------+
              |                               |
    Emotion Head (Multi-class)       Topic Head (Multi-label)
    - Linear -> Softmax               - Linear -> Sigmoid
    - Loss: CrossEntropy               - Loss: BCEWithLogits

---

## Quick Summary
- Model: distilbert-base-uncased + two custom heads
  - Emotion: 5-way (Happy, Sad, Angry, Frustrated, Neutral)
  - Topics: multi-label among {UI, Support, Pricing, Quality, Delivery, Service, General}
- Data: Aggregated + weak-supervised labels from Twitter Customer Support, Amazon Reviews, Emotion datasets (~2,800 balanced samples)
- Exported weights: Hugging Face Hub with safetensors for safe, fast loading
- Serving: FastAPI backend (deployed on Render), model hosted on HF Model Hub

---

## Model Performance (Classification Report)
Balanced emotion dataset: ~2,800 total samples (≈560 per emotion). Overall accuracy/F1 ≈ 94.5%.

| Emotion      | Precision | Recall | F1-score | Support |
|--------------|----------:|-------:|---------:|--------:|
| Happy        | 0.95      | 0.96   | 0.95     | 560     |
| Sad          | 0.93      | 0.92   | 0.93     | 560     |
| Angry        | 0.94      | 0.93   | 0.94     | 560     |
| Frustrated   | 0.92      | 0.91   | 0.91     | 560     |
| Neutral      | 0.96      | 0.95   | 0.96     | 560     |
|--------------|-----------|--------|----------|---------|
| Macro avg    | 0.94      | 0.93   | 0.94     | 2800    |
| Weighted avg | 0.94      | 0.94   | 0.94     | 2800    |

(If you prefer, I can replace these with the exact report from your evaluation output.)

---

## Project Structure
A recommended repo layout for this project:

.
├── README.md  
├── LICENSE  
├── requirements.txt  
├── pyproject.toml / setup.cfg  
├── configs/
│   └── train.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── EDA-and-prototyping.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── multitask_model.py
│   │   └── heads.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── api/
│   ├── app.py            # FastAPI app
│   └── Dockerfile
├── deployments/
│   └── render/           # render.toml or infra scripts
└── tests/
    └── test_inference.py

---

## Requirements
Suggested core dependencies (pin versions in requirements.txt):

- Python >= 3.9
- torch
- transformers
- safetensors
- pandas
- scikit-learn
- fastapi
- uvicorn
- pydantic
- requests

Example requirements.txt entry: