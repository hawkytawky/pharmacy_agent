"""Bigram model tab component for the LLM page."""

import logging
from pathlib import Path
from typing import Dict

import plotly.graph_objects as go

import streamlit as st
from models.bigram import generate_text, list_available_models, load_model

logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path("models/training_data")

MODEL_INSIGHTS = {
    "goethe": """
    ⚠️ **Overfitting Alert:** Loss gap of ~0.98 (Train: 2.50 vs Val: 3.48).
    The model is memorizing the small dataset. The high vocabulary size (91 chars)
    dilutes the limited data, preventing it from learning general rules.
    """,
    "shakespear": """
    ✅ **Stable Generalization:** Tiny loss gap of -0.007 (Train: 2.57 vs Val: 2.56).
    The curves track perfectly together. The larger corpus and compact vocabulary (65 chars)
    allow the model to learn robust character patterns without memorizing.
    """,
}


def load_training_text(model_name: str, max_chars: int = 300) -> str:
    """Load the first n characters of the training text.

    Args:
        model_name: Name of the model (matches the .txt filename).
        max_chars: Maximum characters to return.

    Returns:
        First n characters of the training text or error message.
    """
    text_path = TRAINING_DATA_DIR / f"{model_name}.txt"
    if not text_path.exists():
        return f"Training text not found: {text_path}"

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read(max_chars)

    return text + "..." if len(text) == max_chars else text


def render_training_info(checkpoint: Dict, selected_model: str) -> None:
    """Render training information from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary.
        selected_model: Name of the selected model for insights.
    """
    st.subheader("📊 Training Information")

    # Hyperparameters
    hyperparams = checkpoint.get("hyperparameters", {})
    if hyperparams:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Batch Size", hyperparams.get("batch_size", "N/A"))
            st.metric("Block Size", hyperparams.get("block_size", "N/A"))
        with col2:
            st.metric("Max Iterations", hyperparams.get("max_iters", "N/A"))
            st.metric("Eval Interval", hyperparams.get("eval_interval", "N/A"))
        with col3:
            st.metric("Learning Rate", hyperparams.get("learning_rate", "N/A"))
            st.metric("Vocab Size", checkpoint.get("vocab_size", "N/A"))

    # Loss chart
    history = checkpoint.get("training_history", {})
    if history and history.get("steps"):
        st.subheader("📉 Training Loss")

    insight = MODEL_INSIGHTS.get(selected_model)
    if insight:
        st.info(insight)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history["steps"],
                y=history["train_loss"],
                mode="lines+markers",
                name="Train Loss",
                line={"color": "#1f77b4"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=history["steps"],
                y=history["val_loss"],
                mode="lines+markers",
                name="Validation Loss",
                line={"color": "#ff7f0e"},
            )
        )
        fig.update_layout(
            xaxis_title="Training Steps",
            yaxis_title="Loss",
            legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Final loss metrics with gap indicator
        train_loss = history["train_loss"][-1]
        val_loss = history["val_loss"][-1]
        loss_gap = val_loss - train_loss

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Train Loss", f"{train_loss:.4f}")
        with col2:
            st.metric("Final Validation Loss", f"{val_loss:.4f}")
        with col3:
            # Color code the gap
            gap_status = "🟢" if loss_gap < 0.3 else "🟡" if loss_gap < 0.5 else "🔴"
            st.metric("Loss Gap", f"{gap_status} {loss_gap:.4f}")


def render_bigram_tab() -> None:
    """Render the Bigram model tab content."""
    st.markdown(
        """
        The **Bigram Language Model** is a simple character-level model that predicts
        the next character based only on the previous one. It learns character transition
        probabilities from the training text.
        """
    )

    # Get available models
    available_models = list_available_models()

    if not available_models:
        st.warning(
            "⚠️ No trained models found. Train a model first using:\n\n"
            "```bash\n"
            "python -m models.bigram.run --input models/training_data/goethe.txt\n"
            "```"
        )
        return

    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        format_func=lambda x: x.capitalize(),
    )

    if not selected_model:
        return

    # Load model
    try:
        model, encoder, checkpoint = load_model(selected_model)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Training text preview
    st.subheader("📖 Training Text Preview")
    training_text = load_training_text(selected_model)
    st.code(training_text, language=None)

    st.divider()

    # Training information
    render_training_info(checkpoint, selected_model)

    st.divider()

    # Generation section
    st.subheader("✨ Text Generation")
    st.info(
        "Just to clarify, the generated text is not going to be meaningful 😅 The bigram model is very simple and only captures character-level transitions, so the output will be mostly gibberish. But it's the fundament of language models and a fun demonstration of how a language model can generate text based on learned probabilities!"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        num_tokens = st.slider("Number of characters to generate", 50, 500, 200)
    with col2:
        generate_button = st.button(
            "🎲 Generate", type="primary", use_container_width=True
        )

    if generate_button:
        with st.spinner("Generating text..."):
            generated = generate_text(model, encoder, max_tokens=num_tokens)
        st.text_area("Generated Text", generated, height=200)
