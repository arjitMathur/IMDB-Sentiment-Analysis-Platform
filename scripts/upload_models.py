"""Upload trained model artifacts to HuggingFace Hub.

This script uploads all model files from models/ to a HuggingFace model
repository so they can be downloaded by the deployed app.

Prerequisites:
    1. Create a HuggingFace account at https://huggingface.co
    2. Create a new model repository named 'imdb-sentiment-models'
    3. Log in: huggingface-cli login

Usage:
    python scripts/upload_models.py
"""

from huggingface_hub import HfApi
from pathlib import Path

REPO_ID = "arjitMathur/imdb-sentiment-models"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Files to upload per model (excludes training checkpoints)
UPLOAD_MAP = {
    "baseline": [
        "tfidf_vectorizer.joblib",
        "logistic_regression.joblib",
        "metrics.json",
    ],
    "lstm": [
        "sentiment_model.keras",
        "tokenizer.pickle",
        "metrics.json",
    ],
    "transformer_onnx": [
        "model.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "metrics.json",
    ],
}


def main():
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repository: https://huggingface.co/{REPO_ID}")

    for model_name, files in UPLOAD_MAP.items():
        for filename in files:
            local_path = MODEL_DIR / model_name / filename
            if not local_path.exists():
                print(f"  SKIP {model_name}/{filename} (not found)")
                continue

            remote_path = f"{model_name}/{filename}"
            print(f"  Uploading {remote_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  ✅ {remote_path}")

    print(f"\nDone! Models available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
