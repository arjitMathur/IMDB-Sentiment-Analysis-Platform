"""Sentiment Analysis Platform — Streamlit Demo Application.

A polished, professional UI that demonstrates all models with:
    - Model selector dropdown
    - Real-time prediction with confidence visualization
    - Latency display
    - Model comparison table
    - Benchmark metrics dashboard
    - Graceful "not trained yet" states

Run with:
    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports after page config
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

# Add project root to sys.path so Python can find the 'src' package
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.sentiment.registry import ModelRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .metric-card h3 {
        color: #a8b2d1;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #e6f1ff;
        font-size: 2rem;
        font-weight: 700;
    }

    /* Confidence bar */
    .confidence-bar {
        background: #1a1a2e;
        border-radius: 12px;
        overflow: hidden;
        height: 32px;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
        transition: width 0.5s ease;
    }
    .positive-fill { background: linear-gradient(90deg, #00b894, #00cec9); }
    .negative-fill { background: linear-gradient(90deg, #e17055, #d63031); }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Model badge */
    .model-badge {
        display: inline-block;
        background: rgba(108, 92, 231, 0.2);
        color: #a29bfe;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        border: 1px solid rgba(108, 92, 231, 0.3);
    }

    /* Header gradient */
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }

    /* Comparison table */
    .comparison-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State & Registry
# ---------------------------------------------------------------------------
@st.cache_resource
def get_registry() -> ModelRegistry:
    """Cached model registry (survives reruns).

    On cloud deployments, models are downloaded from HuggingFace Hub
    on first launch.
    """
    try:
        from src.sentiment.model_downloader import ensure_models
        with st.spinner("📦 Downloading models on first launch (this only happens once)..."):
            ensure_models()
    except Exception as e:
        st.warning(f"Could not auto-download models: {e}")
    return ModelRegistry()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(registry: ModelRegistry) -> tuple[str, str]:
    """Render sidebar and return (selected_model_key, review_text)."""
    with st.sidebar:
        st.markdown("### 🎯 Configuration")

        # Model selection
        all_models = registry.list_all()
        available = set(registry.list_available())

        model_options = {}
        for key in all_models:
            status = "✅" if key in available else "❌ Not trained"
            model_options[f"{key.upper()} {status}"] = key

        selected_display = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0,
            help="Choose which model to use for predictions.",
        )
        selected_key = model_options[selected_display]

        st.markdown("---")

        # Sample reviews
        st.markdown("### 📝 Sample Reviews")
        samples = {
            "Positive: Amazing movie": "This movie was absolutely amazing! The acting was superb, "
            "the storyline was gripping, and I was on the edge of my seat the entire time. "
            "Definitely one of the best films I've seen this year!",
            "Negative: Terrible film": "What a waste of time. The plot was predictable, the acting was wooden, "
            "and I almost fell asleep halfway through. Save your money and skip this one.",
            "Mixed: Average": "It was okay, I guess. Some parts were interesting but overall it felt "
            "like a generic movie that didn't bring anything new to the table.",
            "Short: One word": "disgusting",
        }

        if st.button("Load Sample ⬆️", use_container_width=True):
            sample_key = st.session_state.get("sample_key", list(samples.keys())[0])
            st.session_state["review_text"] = samples[sample_key]

        sample_key = st.radio("Pick a sample:", list(samples.keys()), label_visibility="collapsed")
        st.session_state["sample_key"] = sample_key

        if st.button("Use This Sample", use_container_width=True):
            st.session_state["review_text"] = samples[sample_key]

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "Multi-model sentiment analysis platform. "
            "Compares **TF-IDF**, **LSTM**, and **DistilBERT** approaches."
        )

    return selected_key, st.session_state.get("review_text", "")


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
def render_prediction(registry: ModelRegistry, model_key: str, review_text: str) -> None:
    """Render the prediction input and results."""
    st.markdown('<p class="header-gradient">🎬 Sentiment Analysis Platform</p>', unsafe_allow_html=True)
    st.markdown("Analyze movie review sentiment with multiple ML models.")
    st.markdown("")

    # Input area
    review = st.text_area(
        "Enter your review:",
        value=review_text,
        height=120,
        placeholder="Type or paste a movie review here...",
        key="review_input",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with col_info:
        st.caption(f"Using model: **{model_key.upper()}**")

    if predict_clicked and review.strip():
        model = registry.load(model_key)

        if model is None:
            st.error(
                f"⚠️ Model **{model_key}** is not trained yet. "
                "Please train it first with `python -m src.sentiment.train`.",
                icon="🚫",
            )
            return

        # Run prediction
        with st.spinner("Analyzing..."):
            result = model.predict(review)

        # Display result
        label = result["label"]
        confidence = result["confidence"]
        pos_prob = result["probabilities"]["positive"]
        neg_prob = result["probabilities"]["negative"]
        latency = result.get("latency_ms", 0)

        emoji = "😊" if label == "positive" else "😞"
        color = "#00cec9" if label == "positive" else "#d63031"
        fill_class = "positive-fill" if label == "positive" else "negative-fill"

        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 3rem;">{emoji}</span>
                    <span style="font-size: 1.8rem; font-weight: 700; color: {color}; margin-left: 0.5rem;">
                        {label.upper()}
                    </span>
                </div>
                <div>
                    <span class="model-badge">{model.name}</span>
                    <span style="color: #a8b2d1; margin-left: 1rem;">⏱️ {latency:.1f}ms</span>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <div style="color: #a8b2d1; margin-bottom: 0.25rem;">Confidence</div>
                <div class="confidence-bar">
                    <div class="confidence-fill {fill_class}" style="width: {confidence*100:.0f}%;">
                        {confidence*100:.1f}%
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive Probability", f"{pos_prob:.4f}")
        with col2:
            st.metric("Negative Probability", f"{neg_prob:.4f}")
        with col3:
            st.metric("Latency", f"{latency:.1f} ms")

    elif predict_clicked:
        st.warning("Please enter a review to analyze.")


def render_comparison(registry: ModelRegistry, review_text: str) -> None:
    """Render side-by-side model comparison."""
    st.markdown("---")
    st.markdown("### 🔄 Model Comparison")
    st.markdown("Compare predictions across all available models.")

    if not review_text or not review_text.strip():
        st.info("Enter a review above to see comparison results.")
        return

    if st.button("⚡ Compare All Models", use_container_width=False):
        available = registry.list_available()
        if not available:
            st.warning("No trained models available for comparison.")
            return

        results = []
        for key in available:
            model = registry.load(key)
            if model:
                result = model.predict(review_text)
                result["model_key"] = key
                result["model_name"] = model.name
                results.append(result)

        if results:
            # Display as columns
            cols = st.columns(len(results))
            for col, res in zip(cols, results):
                with col:
                    emoji = "😊" if res["label"] == "positive" else "😞"
                    st.markdown(f"**{res['model_name']}**")
                    st.markdown(f"### {emoji} {res['label'].upper()}")
                    st.progress(res["confidence"])
                    st.caption(f"Confidence: {res['confidence']:.4f}")
                    st.caption(f"Latency: {res.get('latency_ms', 0):.1f}ms")


def render_benchmark(registry: ModelRegistry) -> None:
    """Render benchmark metrics dashboard."""
    st.markdown("---")
    st.markdown("### 📊 Benchmark Metrics")

    metrics_data = []
    for key in registry.list_all():
        m = registry.get_metrics(key)
        if m:
            metrics_data.append({
                "Model": m.get("model", key),
                "Accuracy": m.get("test_accuracy", m.get("val_accuracy", "N/A")),
                "F1 Score": m.get("test_f1", m.get("val_f1", "N/A")),
                "Training Time (s)": m.get("training_time_seconds", "N/A"),
            })

    if metrics_data:
        st.table(metrics_data)
    else:
        st.info(
            "No benchmark metrics available yet. "
            "Train models with `python -m src.sentiment.train all` to see results here."
        )


# ---------------------------------------------------------------------------
# App Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Main application entry point."""
    registry = get_registry()
    model_key, review_text = render_sidebar(registry)

    # Use text from input area if available, else from sidebar
    render_prediction(registry, model_key, review_text)
    render_comparison(registry, st.session_state.get("review_input", review_text))
    render_benchmark(registry)


if __name__ == "__main__":
    main()
