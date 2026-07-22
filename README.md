# Review Analyzer — Full-Stack Transformer AI App

Fine-tunes DistilBERT with **two heads** on customer reviews:
- **Emotion (single-label, softmax)** → Happy, Angry, Sad, Frustrated, Neutral
- **Topics (multi-label, sigmoid)** → Pricing, UI, Service, Quality, Delivery, Support

## Architecture

```
┌──────────────┐   HTTP    ┌──────────────┐   HTTP    ┌───────────────────┐
│  React (UI)  │ ────────► │ Node/Express │ ────────► │ FastAPI + PyTorch │
│   port 5173  │           │  port 4000   │           │    port 8000      │
└──────────────┘           └──────────────┘           └───────────────────┘
```

## Directory layout

```
review-analyzer/
├── ml/                       # Training code (run once)
│   ├── requirements.txt
│   ├── data_sample.csv
│   └── train.py
├── ml-service/               # FastAPI inference microservice
│   ├── requirements.txt
│   └── main.py
├── backend/                  # Node.js gateway
│   ├── package.json
│   ├── .env.example
│   └── server.js
└── frontend/                 # React + Tailwind UI
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── index.css
        ├── App.jsx
        └── components/
            ├── ReviewForm.jsx
            ├── EmotionCard.jsx
            └── TopicBars.jsx
```

## Setup (three terminals)

### 1) Train the model (once)
```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py            # writes ../ml-service/model/
```

### 2) Start the Python inference service
```bash
cd ml-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3) Start the Node gateway
```bash
cd backend
cp .env.example .env
npm install
npm run start             # listens on :4000
```

### 4) Start the React app
```bash
cd frontend
npm install
npm run dev               # opens :5173
```

## Endpoints

- `POST http://localhost:8000/predict` — `{ "text": "..." }` (internal)
- `POST http://localhost:4000/api/analyze` — `{ "text": "..." }` (public)

Response shape:
```json
{
  "emotion": { "label": "Happy", "score": 0.93,
               "distribution": { "Happy": 0.93, "Angry": 0.01, ... } },
  "topics":  [ { "label": "Delivery", "score": 0.88 }, ... ]
}
```
