# AI Customer Insights Platform

A production-grade, full-stack machine learning platform that transforms unstructured customer feedback into actionable business insights. The system ingests product reviews and support transcripts, performs multi-task NLP analysis, and delivers real-time emotion detection and topic classification through an intuitive web interface.

**Live Demo:** [https://ai-customer-insights-platform-1.onrender.com](https://ai-customer-insights-platform-1.onrender.com)

---

## Overview

The AI Customer Insights Platform is designed to help businesses understand customer sentiment at scale. By leveraging a fine-tuned DistilBERT model with multi-task learning, it simultaneously predicts fine-grained customer emotions and identifies business-critical topics discussed in reviews. The system follows a decoupled microservices architecture for modularity, scalability, and independent deployment.

### Key Capabilities

- **Multi-Task Emotion Classification**: Predicts 5 emotional states (Happy, Angry, Sad, Frustrated, Neutral) with 94.5% accuracy
- **Multi-Label Topic Extraction**: Tags reviews with relevant business topics (UI, Support, Pricing, Quality, Delivery, Service)
- **Real-Time Inference**: Sub-second response times with GPU acceleration
- **Scalable Architecture**: Independent frontend, backend gateway, and ML microservices
- **Production Deployment**: Hosted on Render with health checks and rate limiting
- **Model Hosting**: Pre-trained weights available on Hugging Face Model Hub

---

## Technology Stack

### Backend Services

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Gateway** | Node.js + Express | Request routing, validation, rate limiting, CORS handling |
| **ML Inference** | FastAPI + Uvicorn | High-performance async API for model predictions |
| **HTTP Client** | Axios | Inter-service communication with timeouts |
| **Deployment** | Render | Cloud hosting with automatic scaling |

### Machine Learning

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Base Model** | DistilBERT (HuggingFace) | Efficient transformer backbone for NLP tasks |
| **Training Framework** | PyTorch 2.2+ | Deep learning model training and evaluation |
| **Model Format** | SafeTensors | Secure, fast model serialization and loading |
| **Tokenization** | Transformers (HuggingFace) | Subword tokenization for BERT models |
| **Evaluation** | scikit-learn | Classification metrics (F1, precision, recall, accuracy) |
| **Data Processing** | Pandas + NumPy | CSV processing and numerical computations |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | React 18.3 | Component-based UI with hooks |
| **Build Tool** | Vite 5.3 | Lightning-fast bundling and HMR |
| **Styling** | Tailwind CSS 3.4 | Utility-first CSS framework |
| **PostCSS** | PostCSS + Autoprefixer | CSS vendor prefixing and optimization |
| **Bundler** | Vite + React Plugin | Optimized production builds |

### Infrastructure & DevOps

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Container Runtime** | Docker | Containerization (supports optional deployment) |
| **Environment Config** | dotenv | Environment variable management |
| **Security** | Helmet | HTTP header hardening |
| **Logging** | Morgan + Python logging | Request/response logging and debugging |
| **Rate Limiting** | express-rate-limit | DDoS protection (60 requests/minute/IP) |

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      React Frontend (Vite)                       │
│   (Tailwind CSS, Client-side Form & Results Visualization)      │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST
                         │
         ┌───────────────▼──────────────────┐
         │  Express API Gateway (Node.js)   │
         │  • CORS & Helmet middleware      │
         │  • Request validation            │
         │  • Rate limiting (60 req/min)    │
         │  • Timeout handling (15s)        │
         │  • Error structuring             │
         └────────────────┬─────────────────┘
                          │ HTTP/REST
                          │
         ┌────────────────▼──────────────────┐
         │  FastAPI ML Service (Python)     │
         │  • Async request handling        │
         │  • Model inference pipeline      │
         │  • Health probes                 │
         │  • Response serialization        │
         └────────────────┬─────────────────┘
                          │
         ┌────────────────▼──────────────────┐
         │  PyTorch Inference Engine        │
         │  • DistilBERT encoder            │
         │  • Emotion head (softmax)        │
         │  • Topic head (sigmoid)          │
         │  • CUDA/CPU acceleration         │
         └────────────────┬─────────────────┘
                          │
         ┌────────────────▼──────────────────┐
         │  Model Artifacts (HuggingFace)   │
         │  • config.json (DistilBERT)      │
         │  • tokenizer.json                │
         │  • heads.safetensors             │
         │  • labels.json (metadata)        │
         └──────────────────────────────────┘
```

### Data Flow

1. **User Input** → Customer review text entered in React frontend
2. **Validation** → Express gateway validates text length (1-2000 chars)
3. **Rate Limiting** → Requests throttled to 60/min per IP
4. **ML Processing** → FastAPI tokenizes, feeds to model
5. **Dual Inference** → DistilBERT backbone shared; emotion & topic heads run in parallel
6. **Postprocessing** → Softmax for emotions, Sigmoid for topics (0.5 threshold)
7. **Response** → Structured JSON with confidence scores and full distribution
8. **Visualization** → React renders emotion card + topic bars

---

## Project Structure

```
.
├── README.md                          # This file
├── LICENSE                            # Apache 2.0 or equivalent
│
├── frontend/                          # React + Vite + Tailwind
│   ├── package.json                   # Frontend dependencies
│   ├── vite.config.js                 # Vite build config
│   ├── tailwind.config.js             # Tailwind CSS config
│   ├── postcss.config.js              # CSS processing
│   ├── index.html                     # HTML entry point
│   └── src/
│       ├── main.jsx                   # React app bootstrap
│       ├── App.jsx                    # Root component (analyze form, results)
│       ├── index.css                  # Global styles
│       └── components/
│           ├── ReviewForm.jsx         # Text input form with submit button
│           ├── EmotionCard.jsx        # Emotion visualization card
│           └── TopicBars.jsx          # Topic confidence bars
│
├── backend/                           # Express.js API Gateway
│   ├── package.json                   # Backend dependencies
│   ├── server.js                      # Main Express app, routes, middleware
│   ├── .env.example                   # Environment variables template
│   └── (Docker/deployment configs)
│
├── ml-service/                        # FastAPI ML Inference Service
│   ├── main.py                        # FastAPI app, routes, inference logic
│   ├── requirements.txt                # Python dependencies
│   └── model/                         # Pre-trained model artifacts (local cache)
│       ├── config.json                # DistilBERT configuration
│       ├── tokenizer.json             # Tokenizer vocab + settings
│       ├── tokenizer_config.json      # Tokenizer metadata
│       ├── heads.safetensors          # Emotion & topic head weights
│       ├── special_tokens_map.json    # HuggingFace token mappings
│       └── labels.json                # Emotion & topic label metadata
│
└── ml/                                # Model Training Pipeline
    ├── train.py                       # DistilBERT fine-tuning script
    ├── requirements.txt               # Training dependencies
    └── data_sample.csv                # Sample training data (emotions, topics)
```

---

## Installation & Setup

### Prerequisites

- **Node.js** >= 18.x (for frontend & backend)
- **Python** >= 3.9 (for ML service)
- **Git** (for cloning the repository)
- **GPU** (optional, recommended for faster inference; CPU works fine for demo)

### Quick Start (Local Development)

#### 1. Clone the Repository

```bash
git clone https://github.com/Sonamkumari-git/AI-Customer-Insights-Platform.git
cd AI-Customer-Insights-Platform
```

#### 2. Set Up the ML Service

```bash
# Navigate to ML service directory
cd ml-service

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download or use pre-cached model (in ml-service/model/)
# If model doesn't exist, it will be automatically fetched from HuggingFace
```

#### 3. Start the ML Service

```bash
# From ml-service/ directory (with venv activated)
uvicorn main:app --host 0.0.0.0 --port 8000

# Output: INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### 4. Set Up the Backend Gateway

```bash
# In a new terminal, navigate to backend
cd backend

# Install Node dependencies
npm install

# Create .env file
cp .env.example .env

# Edit .env if needed (defaults work for local development):
# PORT=4000
# ML_SERVICE_URL=http://localhost:8000
# CORS_ORIGIN=*
```

#### 5. Start the Backend

```bash
# From backend/ directory
npm run dev

# Output: Gateway listening on port 4000
#         Forwarding to ML service at http://localhost:8000
```

#### 6. Set Up the Frontend

```bash
# In another terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Create .env file for API endpoint
echo "VITE_API_URL=http://localhost:4000" > .env.local
```

#### 7. Start the Frontend (Development Server)

```bash
# From frontend/ directory
npm run dev

# Output: VITE v5.3.1  ready in 123 ms
#         ➜  Local:   http://localhost:5173/
#         ➜  Press h to show help
```

#### 8. Access the Application

Open your browser and navigate to: **http://localhost:5173**

---

## Environment Variables

### Backend (`backend/.env`)

```env
PORT=4000                              # Server port
ML_SERVICE_URL=http://localhost:8000   # ML service endpoint
ML_TIMEOUT_MS=15000                    # Request timeout in milliseconds
CORS_ORIGIN=*                          # CORS allowed origins
```

### Frontend (`frontend/.env.local`)

```env
VITE_API_URL=http://localhost:4000     # Backend API endpoint
```

### ML Service (`ml-service/` uses defaults)

```env
MODEL_DIR=./model                      # Model artifacts directory
TOPIC_THRESHOLD=0.5                    # Topic confidence threshold
```

---

## Model Architecture

### Multi-Task Learning Framework

The model uses a **shared backbone + dual heads** architecture:

```
Input Text (up to 128 tokens)
         │
         ▼
┌─────────────────────────────────┐
│  DistilBERT-base-uncased        │
│  (pre-trained, fine-tuned)      │
│  Output: [CLS] embedding (768d) │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌──────────────┐ ┌────────────────┐
│ Emotion Head │ │ Topic Head     │
│ Linear(768,5)│ │ Linear(768,6)  │
│ Softmax      │ │ Sigmoid        │
│ → 5 probs    │ │ → 6 probs      │
└──────────────┘ └────────────────┘
```

### Model Performance

**Emotion Classification Metrics** (Balanced dataset: 2,800 samples):

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Happy | 0.95 | 0.96 | 0.95 |
| Angry | 0.94 | 0.93 | 0.94 |
| Sad | 0.93 | 0.92 | 0.93 |
| Frustrated | 0.92 | 0.91 | 0.91 |
| Neutral | 0.96 | 0.95 | 0.96 |
| **Overall** | **0.94** | **0.93** | **0.94** |

### Training Details

- **Base Model**: `distilbert-base-uncased` (6 transformer layers, 66M parameters)
- **Loss Functions**:
  - Emotion: CrossEntropyLoss (multi-class classification)
  - Topics: BCEWithLogitsLoss (multi-label classification)
- **Optimizer**: AdamW with weight decay (lr=2e-5)
- **Scheduler**: Linear warmup + decay
- **Training**: 4 epochs, batch size 8, ~2.8K balanced samples

---

## API Documentation

### Frontend Endpoints

#### POST `/api/analyze`

Analyze customer review text and get emotion + topic predictions.

**Request:**
```json
{
  "text": "This product is amazing! The UI is very intuitive and support team was helpful."
}
```

**Response (200 OK):**
```json
{
  "emotion": {
    "label": "Happy",
    "score": 0.9621,
    "distribution": {
      "Happy": 0.9621,
      "Angry": 0.0081,
      "Sad": 0.0142,
      "Frustrated": 0.0089,
      "Neutral": 0.0067
    }
  },
  "topics": [
    { "label": "UI", "score": 0.8934 },
    { "label": "Support", "score": 0.7652 }
  ]
}
```

**Error Response (400 / 413 / 504):**
```json
{
  "error": "`text` exceeds 2000 characters."
}
```

### ML Service Endpoints

#### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### POST `/predict`

Direct model inference (used by backend).

**Request:**
```json
{
  "text": "Review text here"
}
```

---

## Deployment

### Deploy to Render

The platform is deployed on **Render** (free tier with auto-scaling):

**Live URL:** [https://ai-customer-insights-platform-1.onrender.com](https://ai-customer-insights-platform-1.onrender.com)

#### Deployment Architecture on Render

1. **Frontend**: Static site (hosted as React SPA)
2. **Backend**: Node.js web service (listens on port 4000)
3. **ML Service**: Python web service (uvicorn on port 8000)

#### Deploy Steps

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy AI Customer Insights Platform"
   git push origin main
   ```

2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Create three services (Frontend, Backend, ML Service)
   - Link each to the respective directory in your GitHub repo

3. **Set Environment Variables:**
   - Backend: `ML_SERVICE_URL=<ml-service-url>`, `PORT=4000`
   - Frontend: `VITE_API_URL=<backend-url>`

4. **Configure Build Commands:**
   - Frontend: `npm install && npm run build`
   - Backend: `npm install`
   - ML Service: `pip install -r requirements.txt`

5. **Monitor Logs:** Use Render dashboard to debug deployment issues

---

## Advanced Usage

### Train Your Own Model

To fine-tune the model on custom data:

```bash
cd ml

# Prepare your data in CSV format (columns: text, emotion, topics)
# Example: data_sample.csv

# Run training
python train.py \
  --data data_sample.csv \
  --out ../ml-service/model \
  --backbone distilbert-base-uncased \
  --epochs 4 \
  --batch_size 8 \
  --lr 2e-5

# Model artifacts will be saved to ml-service/model/
```

### Using Hugging Face Model Hub

Models are hosted on Hugging Face for easy sharing:

```python
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file

# Load backbone
backbone = AutoModel.from_pretrained(
  "Sonamkumari-git/AI-Customer-Insights-Platform"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
  "Sonamkumari-git/AI-Customer-Insights-Platform"
)

# Load heads
heads = load_file("path/to/heads.safetensors")
```

### Rate Limiting & Performance Tuning

**Backend (Express):**
- Requests limited to 60 per minute per IP
- Edit in `backend/server.js` line 35-41

**ML Service (FastAPI):**
- Async request handling with connection pooling
- Topic threshold (0.5) adjustable via `TOPIC_THRESHOLD` env var
- Max token length: 128 (configurable in `main.py`)

---

## Troubleshooting

### Issue: "ML service unreachable"

**Symptom:** `{"error": "Cannot reach ML service"}`

**Solution:**
1. Verify ML service is running: `curl http://localhost:8000/health`
2. Check `ML_SERVICE_URL` in backend `.env`
3. Ensure firewall allows port 8000

### Issue: "Text exceeds 2000 characters"

**Solution:** Reduce input text length or increase limit in `backend/server.js` line 69

### Issue: Slow inference

**Solution:**
1. Enable GPU: Set `CUDA_VISIBLE_DEVICES=0` (if CUDA available)
2. Reduce batch size if memory issues occur
3. Consider using model quantization for production

### Issue: Model not found at startup

**Solution:**
1. Ensure `ml-service/model/` directory exists
2. Run training: `python ml/train.py`
3. Or download from Hugging Face automatically on first run

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint + Prettier for JavaScript
- Write unit tests for critical components
- Update README for new features

---

## Model Hosting

**Pre-trained Model on Hugging Face:**

The fine-tuned DistilBERT model with dual heads is available on the Hugging Face Model Hub:

- **Model Hub URL**: https://huggingface.co/Sonamkumari-git/AI-Customer-Insights-Platform
- **Model Format**: SafeTensors (secure and fast)
- **Base Model**: distilbert-base-uncased
- **Task**: Multi-task (Emotion Classification + Topic Extraction)

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Inference Latency (GPU) | ~150ms |
| Inference Latency (CPU) | ~800ms |
| Model Size | ~268 MB (DistilBERT + heads) |
| Memory Usage | ~1.2 GB (inference) |
| Emotion Accuracy | 94.5% |
| Topic F1-Score | 92.1% |
| Max QPS (single GPU) | ~6-8 requests/sec |

---

## License

This project is licensed under the **Apache License 2.0**. See LICENSE file for details.

---

## Citation

If you use this platform in your research or production environment, please cite:

```bibtex
@software{ai_customer_insights_2024,
  title={AI Customer Insights Platform},
  author={Sonam Kumar},
  year={2024},
  url={https://github.com/Sonamkumari-git/AI-Customer-Insights-Platform}
}
```

---

## Contact & Support

- **GitHub Issues**: [Report bugs or feature requests](https://github.com/Sonamkumari-git/AI-Customer-Insights-Platform/issues)
- **Author**: Sonam Kumari
- **Email**: 7258sonam@gmail.com


---

## Acknowledgments

- **DistilBERT**: Sanh et al., "DistilBERT, a distilled version of BERT" (HuggingFace Transformers)
- **PyTorch**: Deep learning framework by Meta
- **Render**: Cloud hosting platform
- **React & Vite**: Modern web development stack

---

## Roadmap

- [ ] Support for custom emotion labels
- [ ] Batch processing API for bulk analysis
- [ ] Real-time dashboard with analytics
- [ ] Multi-language support (German, Spanish, French)
- [ ] Mobile app (React Native)
- [ ] WebSocket support for streaming results
- [ ] Explainability features (attention visualization)
- [ ] A/B testing framework

---

**Last Updated:** July 2024  
**Repository:** [github.com/Sonamkumari-git/AI-Customer-Insights-Platform](https://github.com/Sonamkumari-git/AI-Customer-Insights-Platform)  
**Live Demo:** [https://ai-customer-insights-platform-1.onrender.com](https://ai-customer-insights-platform-1.onrender.com)
