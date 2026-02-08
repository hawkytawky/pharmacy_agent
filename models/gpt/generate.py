"""Text generation for GPT Language Model."""

from pathlib import Path
from typing import Dict, Tuple

from models.common.data import TextEncoder
from models.common.generate import generate_text as _generate_text
from models.common.generate import list_available_models as _list_available_models
from models.common.generate import load_checkpoint
from models.gpt.model import GPTLanguageModel
from models.model_config import DEVICE

CHECKPOINT_DIR = Path("models/gpt/checkpoints")
MODEL_PREFIX = "gpt"


def list_available_models() -> list[str]:
    """List all available trained GPT models."""
    return _list_available_models(CHECKPOINT_DIR, MODEL_PREFIX)


def load_model(model_name: str) -> Tuple[GPTLanguageModel, TextEncoder, Dict]:
    """Load a trained GPT model from checkpoint.

    Args:
        model_name: Name of the model (e.g., 'goethe', 'shakespeare').

    Returns:
        Tuple of (model, encoder, checkpoint_data).
    """
    checkpoint = load_checkpoint(CHECKPOINT_DIR, MODEL_PREFIX, model_name)

    # Recreate encoder from checkpoint
    encoder = TextEncoder.from_checkpoint(checkpoint)

    # Get hyperparameters from checkpoint
    hyperparams = checkpoint.get("hyperparameters", {})

    # Recreate model with saved hyperparameters
    model = GPTLanguageModel(
        vocab_size=encoder.vocab_size,
        n_embed=hyperparams.get("n_embed", 384),
        n_heads=hyperparams.get("n_heads", 6),
        n_layer=hyperparams.get("n_layer", 6),
        block_size=hyperparams.get("block_size", 256),
        dropout=hyperparams.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, encoder, checkpoint


def generate_text(
    model: GPTLanguageModel, encoder: TextEncoder, max_tokens: int = 300
) -> str:
    """Generate text using a trained GPT model."""
    return _generate_text(model, encoder, max_tokens)


def generate_from_checkpoint(model_name: str, max_tokens: int = 300) -> str:
    """Load model and generate text in one call."""
    model, encoder, _ = load_model(model_name)
    return generate_text(model, encoder, max_tokens)
