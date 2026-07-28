# Architecture

## Overview

The Sentiment Analysis Platform follows a layered architecture separating data handling, model logic, serving, and infrastructure.

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                     │
│              Streamlit UI  ←→  FastAPI Backend            │
├─────────────────────────────────────────────────────────┤
│                    Service Layer                         │
│              Model Registry  ←  Configuration            │
├─────────────────────────────────────────────────────────┤
│                    Model Layer                           │
│     Baseline (TF-IDF)  │  LSTM (Keras)  │  DistilBERT   │
│         scikit-learn    │  TensorFlow     │  ONNX Runtime │
├─────────────────────────────────────────────────────────┤
│                    Data Layer                            │
│      HuggingFace Datasets  →  Preprocessing Pipeline     │
└─────────────────────────────────────────────────────────┘
```

## Design Decisions

### 1. Protocol-Based Model Interface

**Decision:** Use Python's `Protocol` (structural subtyping) instead of abstract base classes.

**Rationale:** Models don't need to inherit from a shared base class. Any object with `predict()`, `load()`, and `name` automatically satisfies the interface. This is more Pythonic and avoids tight coupling.

### 2. Centralized Preprocessing

**Decision:** All text cleaning happens in `preprocessing.py` — one function, used everywhere.

**Rationale:** The original project had different preprocessing at train time (none) vs. inference time (just lowercasing). This train/serve skew silently degraded accuracy. A single `clean_text()` function eliminates this entire class of bugs.

### 3. ONNX for Transformer Deployment

**Decision:** Export DistilBERT to ONNX rather than deploying with PyTorch.

**Rationale:** PyTorch is ~2GB installed. ONNX Runtime is ~50MB and provides equivalent inference speed. This keeps the Docker image under 1.5GB instead of 3.5GB+.

### 4. Lazy Model Loading

**Decision:** Models are imported and loaded on-demand, not at application startup.

**Rationale:** Importing TensorFlow takes 5-10 seconds. If the user only wants the baseline model, they shouldn't pay the TF import cost. The registry uses `importlib.import_module()` to defer imports.

### 5. Separation of Training and Inference

**Decision:** Each model has separate training (`model.py`) and inference (`model_infer.py`) modules.

**Rationale:** Training requires heavy dependencies (full TF, PyTorch, datasets). Inference needs only the model's runtime. This separation enables `requirements.txt` (lightweight) vs `requirements-train.txt` (heavy).

### 6. HuggingFace Datasets Over Kaggle

**Decision:** Replace `kagglehub` with `datasets.load_dataset("imdb")`.

**Rationale:** Kaggle requires API credentials (~/.kaggle/kaggle.json), which is a barrier for anyone cloning the repo. HuggingFace datasets downloads anonymously and caches locally.

## Data Flow

```
1. Raw text         "Great movie!<br /><br />Loved it &amp; the acting."
                                    ↓
2. clean_text()     "great movie loved it the acting"
                                    ↓
3. Model-specific   TF-IDF vectorize  │  Keras tokenize+pad  │  HF tokenize
                                    ↓
4. Inference        LogReg predict    │  LSTM predict          │  ONNX session.run
                                    ↓
5. Output           {"label": "positive", "confidence": 0.94, ...}
```

## Security Considerations

- **API Key Auth:** Optional, controlled by `API_KEY` env var. Disabled by default for demo.
- **Input Validation:** Pydantic enforces max 5000 chars per review, max 32 texts per batch.
- **Non-Root Docker:** Container runs as `appuser`, not root.
- **No Committed Secrets:** `.gitignore` excludes `.env`, `kaggle.json`, model binaries.
- **Pickle Risk:** The LSTM tokenizer uses pickle serialization. For production, consider migrating to JSON-based tokenizer serialization.
