"""Text generation using trained Bigram Language Model."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

from models.bigram.data import TextEncoder
from models.bigram.model import BigramLanguageModel
from models.model_config import DEVICE

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("models/bigram/checkpoints")


def list_available_models() -> list[str]:
    """List all available trained models.

    Returns:
        List of model names (without 'bigram_' prefix and '.pt' suffix).
    """
    if not CHECKPOINT_DIR.exists():
        return []

    models = []
    for f in CHECKPOINT_DIR.glob("bigram_*.pt"):
        # Extract name: bigram_goethe.pt -> goethe
        name = f.stem.replace("bigram_", "")
        models.append(name)
    return sorted(models)


def load_model(model_name: str) -> Tuple[BigramLanguageModel, TextEncoder, Dict]:
    """Load a trained model from checkpoint.

    Args:
        model_name: Name of the model (e.g., 'goethe', 'shakespeare').

    Returns:
        Tuple of (model, encoder, checkpoint_data).

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
    """
    checkpoint_path = CHECKPOINT_DIR / f"bigram_{model_name}.pt"

    if not checkpoint_path.exists():
        available = list_available_models()
        raise FileNotFoundError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

    logger.info(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Recreate encoder from saved mappings
    encoder = TextEncoder.__new__(TextEncoder)
    encoder.vocab_size = checkpoint["vocab_size"]
    encoder.stoi = checkpoint["stoi"]
    encoder.itos = checkpoint["itos"]

    # Recreate model
    model = BigramLanguageModel(encoder.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    logger.info(f"Model loaded successfully. Vocab size: {encoder.vocab_size}")

    return model, encoder, checkpoint


def generate_text(
    model: BigramLanguageModel, encoder: TextEncoder, max_tokens: int = 300
) -> str:
    """Generate text using the trained model.

    Args:
        model: Trained BigramLanguageModel.
        encoder: TextEncoder for decoding output.
        max_tokens: Number of tokens to generate.

    Returns:
        Generated text string.
    """
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_indices = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    return encoder.decode(generated_indices)


def generate_from_checkpoint(model_name: str, max_tokens: int = 300) -> str:
    """Convenience function to load model and generate text.

    Args:
        model_name: Name of the model (e.g., 'goethe', 'shakespeare').
        max_tokens: Number of tokens to generate.

    Returns:
        Generated text string.
    """
    model, encoder, _ = load_model(model_name)
    return generate_text(model, encoder, max_tokens)
