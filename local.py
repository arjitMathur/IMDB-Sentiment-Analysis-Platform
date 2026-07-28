import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATASET_SLUG = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
CSV_FILENAME = "IMDB Dataset.csv"
NUM_WORDS = 5000
MAX_LEN = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
TEST_SIZE = 0.20
VAL_TEST_SPLIT = 0.50
EPOCHS = 9
BATCH_SIZE = 128
PATIENCE = 3
MODEL_SAVE_PATH = "sentiment_imdb_model.keras"
TOKENIZER_SAVE_PATH = "tokenizer.pickle"

def download_and_load_data(dataset_slug, csv_filename):
    print(f"Downloading dataset: {dataset_slug}...")
    try:
        download_path = kagglehub.dataset_download(dataset_slug)
        print(f"Dataset downloaded to folder: {download_path}")
        csv_file_path = os.path.join(download_path, csv_filename)

        if not os.path.exists(csv_file_path):
            alternative_path = os.path.join(download_path, dataset_slug.split('/')[-1], csv_filename)
            if os.path.exists(alternative_path):
                 csv_file_path = alternative_path
                 print(f"Found dataset at alternative path: {csv_file_path}")
            else:
                 raise FileNotFoundError(f"{csv_filename} not found in the downloaded path: {download_path} or common subdirectories.")

        print(f"Loading dataset from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        print(f"Initial dataset shape: {df.shape}")
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"Error during dataset download or loading: {e}")
        print("Please ensure Kaggle API credentials (e.g., ~/.kaggle/kaggle.json) are configured and the dataset slug is correct.")
        raise

def preprocess_data(df, test_size=TEST_SIZE, val_test_split=VAL_TEST_SPLIT, num_words=NUM_WORDS, max_len=MAX_LEN):
    print("\nPreprocessing data...")
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    print("Sentiment value counts after mapping:")
    print(df["sentiment"].value_counts())

    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['sentiment'])
    val_df, test_df = train_test_split(temp_df, test_size=val_test_split, random_state=42, stratify=temp_df['sentiment'])
    print(f"Data split sizes -> Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_df["review"])

    with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["review"]), maxlen=max_len)
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_df["review"]), maxlen=max_len)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_df["review"]), maxlen=max_len)

    Y_train = train_df["sentiment"].values
    Y_val = val_df["sentiment"].values
    Y_test = test_df["sentiment"].values

    print("Data preprocessing complete.")
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, tokenizer

def build_model(num_words=NUM_WORDS, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE):
    print("\nBuilding model...")
    model = Sequential()
    model.add(Embedding(input_dim=num_words + 1, output_dim=embedding_dim))
    model.add(LSTM(units=lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("Model built and compiled.")
    model.summary()
    return model

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE):
    print("\nStarting model training...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    print("Model training complete.")
    return history

def evaluate_model(model, X_test, Y_test):
    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Model Test Loss: {loss:.4f}")
    print(f"Model Test Accuracy: {accuracy:.4f}")
    return loss, accuracy

def plot_history(history):
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linestyle='--', marker='x')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--', marker='x')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_test, Y_test):
    print("\nGenerating confusion matrix...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32")

    cm = confusion_matrix(Y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    plt.show()

def predict_sentiment(review, model, tokenizer, max_len=MAX_LEN):
    if not review:
        return "neutral", 0.5

    review_processed = review.lower().strip()
    sequence = tokenizer.texts_to_sequences([review_processed])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    try:
        prediction = model.predict(padded_sequence, verbose=0)
        prob = prediction[0][0]
        sentiment = "positive" if prob > 0.5 else "negative"
        return sentiment, float(prob)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "error", 0.0

if __name__ == "__main__":
    imdb_df = download_and_load_data(DATASET_SLUG, CSV_FILENAME)

    if imdb_df is not None:
        X_train, X_val, X_test, Y_train, Y_val, Y_test, tokenizer = preprocess_data(imdb_df)
        model = build_model()
        history = train_model(model, X_train, Y_train, X_val, Y_val)
        loss, accuracy = evaluate_model(model, X_test, Y_test)
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved successfully to {MODEL_SAVE_PATH}.")

        plot_history(history)
        plot_confusion_matrix(model, X_test, Y_test)

        print("\n--- Testing Predictions ---")
        try:
            loaded_model = load_model(MODEL_SAVE_PATH)
            with open(TOKENIZER_SAVE_PATH, 'rb') as handle:
                loaded_tokenizer = pickle.load(handle)
            print("Model and tokenizer loaded for prediction testing.")

            reviews_to_test = [
                "This movie was amazing, I loved it so much!",
                "This movie was average for others, but I really loved it!",
                "What a waste of time, the acting was terrible.",
                "It was okay, not great but not bad either.",
                "disgusting",
                ""
            ]

            for review in reviews_to_test:
                sentiment, confidence = predict_sentiment(review, loaded_model, loaded_tokenizer)
                print(f"Review: '{review}'")
                print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})\n")

        except FileNotFoundError:
            print("Error: Could not load saved model or tokenizer for prediction test.")
        except Exception as e:
            print(f"An error occurred during prediction testing: {e}")

    else:
        print("Halting script execution due to data loading failure.")