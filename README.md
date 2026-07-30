# 🎬 IMDB Sentiment Analysis Platform

A **production-grade** machine learning platform for movie review sentiment analysis, featuring multiple model architectures, a REST API, and an interactive dashboard.

> Built to demonstrate end-to-end ML engineering: from data pipelines and model training to serving, testing, and containerized deployment.

---

## ✨ Features

| Feature | Description |
|---|---|
| **3 Model Architectures** | TF-IDF Baseline, Keras LSTM, DistilBERT Transformer (ONNX) |
| **FastAPI Backend** | Production REST API with `/predict`, `/batch`, `/health`, `/metrics` |
| **Streamlit Dashboard** | Interactive UI with model comparison, confidence visualization |
| **115 Unit Tests** | Comprehensive test coverage across all components |
| **CI/CD Pipeline** | GitHub Actions for linting, formatting, and test execution |
| **Docker Ready** | Multi-stage Dockerfile + docker-compose for one-command deployment |
| **Centralized Preprocessing** | Single `clean_text()` pipeline eliminates train/serve skew |
| **Model Registry** | Lazy-loading, caching, and graceful fallback for all models |

---

## 🏗️ Architecture

```
├── api/                    # FastAPI backend
│   ├── main.py             # App entrypoint with middleware
│   ├── routes/             # Versioned API endpoints
│   └── schemas.py          # Pydantic request/response models
├── src/sentiment/          # Core ML package
│   ├── config.py           # Pydantic-settings configuration
│   ├── data.py             # HuggingFace data pipeline
│   ├── preprocessing.py    # Centralized text cleaning
│   ├── registry.py         # Model discovery & caching
│   ├── train.py            # Unified training CLI
│   └── models/
│       ├── baseline.py     # TF-IDF + Logistic Regression
│       ├── lstm.py         # Keras LSTM (training)
│       ├── lstm_infer.py   # LSTM inference wrapper
│       ├── transformer.py  # DistilBERT fine-tuning
│       └── transformer_infer.py  # ONNX inference (no PyTorch needed)
├── streamlit_app/          # Interactive dashboard
├── tests/                  # 115 unit tests
├── docs/                   # Architecture & model documentation
├── Dockerfile              # Multi-stage production build
└── docker-compose.yml      # One-command deployment
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/arjitMathur/IMDB-Sentiment-Analysis-Platform.git
cd IMDB-Sentiment-Analysis-Platform
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train baseline (fast, ~30 seconds)
python -m src.sentiment.train baseline

# Train LSTM (~5 minutes on CPU)
python -m src.sentiment.train lstm

# Train all CPU models
python -m src.sentiment.train all
```

For the **Transformer** model (requires GPU):
```bash
pip install -r requirements-train.txt
python -m src.sentiment.models.transformer
```

### 3. Run the API

```bash
python -m uvicorn api.main:app --reload
# API docs at http://localhost:8000/docs
```

### 4. Run the Dashboard

```bash
streamlit run streamlit_app/app.py
# Dashboard at http://localhost:8501
```

---

## 🔌 API Usage

### Single Prediction
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great film!", "Terrible waste of time."]}'
```

### Response Format
```json
{
  "label": "positive",
  "confidence": 0.97,
  "probabilities": {"positive": 0.97, "negative": 0.03},
  "model": "baseline",
  "latency_ms": 2.4
}
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run fast tests only (skip data download)
python -m pytest tests/ -m "not slow"

# With coverage
python -m pytest tests/ --cov=src --cov=api
```

---

## 🐳 Docker

```bash
# Build and run everything
docker-compose up --build

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## 📊 Model Comparison

| Model | Accuracy | F1 Score | Inference Speed |
|---|---|---|---|
| **TF-IDF + LogReg** | ~88% | ~0.88 | < 5ms |
| **Keras LSTM** | ~87% | ~0.87 | ~50ms |
| **DistilBERT (ONNX)** | ~93% | ~0.93 | ~100ms |

---

## 🔧 Configuration

Configuration is managed via environment variables (see `.env.example`):

```env
MODEL_DIR=models
DEFAULT_MODEL=baseline
LOG_LEVEL=INFO
API_KEY=              # Optional: set for API authentication
```

---

## 📚 Documentation

- [Architecture & Design Decisions](docs/ARCHITECTURE.md)
- [Model Card](docs/MODEL_CARD.md)

---

## 🛠️ Tech Stack

- **ML**: TensorFlow/Keras, scikit-learn, HuggingFace Transformers, ONNX Runtime
- **API**: FastAPI, Pydantic, Uvicorn
- **Frontend**: Streamlit
- **Testing**: Pytest (115 tests)
- **CI/CD**: GitHub Actions
- **Containerization**: Docker, docker-compose
- **Code Quality**: Ruff (linting + formatting)

---

## License

MIT
