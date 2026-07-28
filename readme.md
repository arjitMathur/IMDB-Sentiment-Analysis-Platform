
# 🎬 Predicting Movie Review Sentiments with Machine Learning: An IMDb Case Study

This project leverages machine learning and deep learning—specifically an LSTM-based neural network—to predict whether a given IMDb movie review expresses a **positive** or **negative** sentiment. It's a real-world application of Natural Language Processing (NLP) commonly used in opinion mining, customer feedback analysis, and recommendation systems.

---

## 📌 Problem Statement

To develop a binary text classification model that categorizes IMDb movie reviews as either **positive (1)** or **negative (0)**, using deep learning techniques.

---

## 📁 Project Structure

```
IMDb-Sentiment-Analysis/
├── data/                # Raw and preprocessed datasets
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── src/                 # Modular Python scripts
├── images/              # Word clouds, graphs, confusion matrix
├── requirements.txt     # Project dependencies
├── local.py             # Script for local execution
└── README.md            # Project documentation
```

---

## 📊 Dataset

- **Source**: [IMDb 50K Movie Reviews - Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Total Samples**: 50,000  
  - 25,000 positive  
  - 25,000 negative  
- **Format**: Each row contains:
  - `review`: Text of the review  
  - `sentiment`: Label (`positive` or `negative`)

---

## 🔍 Data Preprocessing & Exploration

- Converted sentiment labels to binary:  
  - `positive` → 1  
  - `negative` → 0  
- Applied stratified train-validation-test split (80%-10%-10%)
- Visualized word frequencies with word clouds
- Removed HTML tags, punctuation, stopwords, etc.

---

## 🧠 Model Architecture

Implemented using **TensorFlow & Keras**:

```python
model = Sequential()
model.add(Embedding(input_dim=5001, output_dim=128))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))
```

- **Embedding Layer**: Converts words into dense vector representations  
- **LSTM Layer**: Captures temporal dependencies in text  
- **Dense Output**: Sigmoid activation for binary classification  

---

## ⚙️ Training Details

- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Batch Size**: 128  
- **Epochs**: 9 (with early stopping)

---

## 📈 Model Performance

| Metric             | Value     |
|--------------------|-----------|
| **Validation Accuracy** | 88.04%    |
| **Test Accuracy**        | ~87%      |

**Confusion Matrix:**
- ✅ True Positives: 2,245  
- ✅ True Negatives: 2,159  
- ❌ False Positives: 341  
- ❌ False Negatives: 255  

---

## 🔎 Sample Predictions

- `"This movie was amazing!"` → ✅ Positive (Confidence: 0.97)  
- `"This movie was average for others, but I really loved it!"` → ✅ Positive (Confidence: 0.86)  
- `"disgusting"` → ❌ Negative (Confidence: 0.22)

---

## 💡 Key Takeaways

- Clean, balanced datasets significantly improve model results  
- LSTM layers are effective for sentiment analysis on sequence data  
- Confidence scores help in evaluating ambiguous predictions

---

## ⚠️ Limitations

- Struggles with sarcasm, irony, and mixed sentiments  
- Model performance may not transfer well to other domains (e.g., product reviews)

---

## 💻 Running the Project

### ▶️ On Google Colab

1. Open Google Colab
2. Upload `IMDB_Sentiment_Analysis.ipynb`
3. Click **Runtime > Run All** to execute the notebook

### 💻 Locally

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python local.py
   ```

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow & Keras**
- **Pandas**, **NumPy**
- **NLTK**, **re** (for text preprocessing)
- **Matplotlib**, **Seaborn** (for visualizations)

---

## 📄 License

This project is intended for academic and learning purposes.

---

## 📬 Contact

For questions, feedback, or collaboration, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/arjit-mathur-48b895353/).

---
