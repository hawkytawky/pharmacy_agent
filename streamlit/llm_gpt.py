"""GPT model tab component for the LLM page."""

import logging
from pathlib import Path
from typing import Dict

import plotly.graph_objects as go

import streamlit as st
from models.gpt import generate_text, list_available_models, load_model

logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path("models/training_data")

# Model-specific insights for different training texts
MODEL_INSIGHTS = {
    "goethe": """
    ⚠️ **Critical Overfitting:** Massive loss gap of 2.70 (Train: 1.54 vs Val: 4.24).
    **Generation Consequence:** The model acts like a photocopier. It will likely regurgitate
    exact training phrases verbatim, but output complete nonsense on prompts it hasn't seen.
    """,
    "shakespear": """
    ✅ **Healthy Generalization:** Small loss gap of 0.17 (Train: 1.63 vs Val: 1.81).
    **Generation Consequence:** The model acts like an author. It understands the style's rules
    and can invent new, coherent sentences that never existed in the original text.
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


def render_model_insight(model_name: str) -> None:
    """Render model-specific insight message.

    Args:
        model_name: Name of the selected model.
    """
    insight = MODEL_INSIGHTS.get(model_name)
    if insight:
        st.info(insight)


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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Batch Size", hyperparams.get("batch_size", "N/A"))
            st.metric("Block Size", hyperparams.get("block_size", "N/A"))
        with col2:
            st.metric("Max Iterations", hyperparams.get("max_iters", "N/A"))
            st.metric("Learning Rate", hyperparams.get("learning_rate", "N/A"))
        with col3:
            st.metric("Embedding Dim", hyperparams.get("n_embed", "N/A"))
            st.metric("Attention Heads", hyperparams.get("n_heads", "N/A"))
        with col4:
            st.metric("Layers", hyperparams.get("n_layer", "N/A"))
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


def render_gpt_tab() -> None:
    """Render the GPT model tab content."""
    st.markdown(
        """
        The **GPT (Generative Pre-trained Transformer)** is a character-level language model
        that uses **self-attention** to capture long-range dependencies in text. Unlike the
        simple Bigram model, GPT can consider a context of multiple characters when predicting
        the next one.

        **Architecture highlights:**
        - Multi-head self-attention for context awareness
        - Feed-forward neural networks for transformation
        - Layer normalization and residual connections
        - Positional embeddings to understand character order
        """
    )

    # Get available models
    available_models = list_available_models()

    if not available_models:
        st.warning(
            "⚠️ No trained models found. Train a model first using:\n\n"
            "```bash\n"
            "python -m models.gpt.run --input models/training_data/shakespear.txt\n"
            "```"
        )
        return

    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        format_func=lambda x: x.capitalize(),
        key="gpt_model_select",
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
        """The GPT model generates more coherent text than the Bigram model because it can
        consider context. However, at the character level with limited training data,
        the output will still be imperfect. You might see recognizable words and
        occasionally grammatical structures!"""
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        num_tokens = st.slider(
            "Number of characters to generate", 50, 500, 200, key="gpt_num_tokens"
        )
    with col2:
        generate_button = st.button(
            "🎲 Generate", type="primary", use_container_width=True, key="gpt_generate"
        )

    if generate_button:
        with st.spinner("Generating text..."):
            generated = generate_text(model, encoder, max_tokens=num_tokens)
        st.text_area("Generated Text", generated, height=200, key="gpt_generated_text")
