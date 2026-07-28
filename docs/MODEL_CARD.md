# Model Card

## Overview

This platform implements three sentiment analysis models of increasing complexity. All models are trained on the IMDB movie review dataset and share identical preprocessing.

---

## Model 1: TF-IDF + Logistic Regression (Baseline)

### Description
A classical machine learning approach using term frequency-inverse document frequency features with a logistic regression classifier. Serves as the performance floor.

### Training Data
- **Dataset:** IMDB Movie Reviews (50,000 samples, balanced)
- **Source:** HuggingFace `datasets` library
- **Split:** 22,500 train / 2,500 val / 25,000 test

### Architecture
- TF-IDF Vectorizer (up to 50,000 features, unigrams + bigrams)
- Logistic Regression (C=1.0, LBFGS solver)

### Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | ~89% |
| Test F1 | ~89% |
| Inference Latency | < 1ms |
| Model Size | ~50MB |

### Limitations
- No understanding of word order or context
- Cannot handle unseen vocabulary well
- Limited by bag-of-words representation

---

## Model 2: LSTM

### Description
A deep learning model using learned word embeddings and a Long Short-Term Memory recurrent layer. Captures sequential patterns in text.

### Architecture
```
Embedding(5001, 128) → LSTM(64, dropout=0.2) → Dense(1, sigmoid)
```
- ~700K parameters
- Binary crossentropy loss, Adam optimizer
- Early stopping on validation loss (patience=3)

### Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | ~87% |
| Inference Latency | ~50ms |
| Model Size | ~10MB |

### Limitations
- Keras Tokenizer has a fixed vocabulary (5000 words)
- Sequential processing limits throughput
- No attention mechanism

---

## Model 3: DistilBERT (ONNX)

### Description
A fine-tuned DistilBERT transformer model, exported to ONNX for lightweight deployment. Represents the state-of-the-art approach.

### Architecture
- Base: `distilbert-base-uncased` (66M parameters)
- Fine-tuned with HuggingFace Trainer
- Exported to ONNX for inference without PyTorch

### Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | ~93% |
| Inference Latency | ~30ms |
| Model Size | ~250MB (ONNX) |

### Limitations
- Requires GPU for training (use Colab/Kaggle)
- Larger model size than alternatives
- ONNX export may lose minor precision

---

## Ethical Considerations

- **Training Data Bias:** IMDB reviews are English-language movie reviews. The model may not generalize to other domains (products, restaurants) or languages.
- **Sarcasm & Irony:** All models struggle with sarcastic reviews where surface sentiment differs from intended meaning.
- **Fairness:** The models have not been evaluated for demographic bias in predictions.
- **Intended Use:** Academic demonstration and portfolio project. Not intended for production content moderation without further validation.

---

## Preprocessing

All models share a single preprocessing pipeline (`preprocessing.py`):

1. Remove HTML tags (`<br />`, `<p>`, etc.)
2. Remove HTML entities (`&amp;`, `&lt;`, etc.)
3. Remove URLs
4. Lowercase
5. Remove non-alphanumeric characters
6. Collapse whitespace

This ensures **train/inference parity** — a critical requirement for reliable ML systems.
