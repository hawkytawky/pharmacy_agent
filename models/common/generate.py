"""Common generation utilities for language models."""

import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from models.common.data import TextEncoder
from models.model_config import DEVICE

logger = logging.getLogger(__name__)


def list_available_models(checkpoint_dir: Path, prefix: str) -> List[str]:
    """List all available trained models in a checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoints directory.
        prefix: Model prefix (e.g., 'bigram', 'gpt').

    Returns:
        List of model names (without prefix and '.pt' suffix).
    """
    if not checkpoint_dir.exists():
        return []

    models = []
    for f in checkpoint_dir.glob(f"{prefix}_*.pt"):
        # Extract name: bigram_goethe.pt -> goethe
        name = f.stem.replace(f"{prefix}_", "")
        models.append(name)
    return sorted(models)


def load_checkpoint(checkpoint_dir: Path, prefix: str, model_name: str) -> Dict:
    """Load checkpoint data from file.

    Args:
        checkpoint_dir: Path to checkpoints directory.
        prefix: Model prefix (e.g., 'bigram', 'gpt').
        model_name: Name of the model (e.g., 'goethe').

    Returns:
        Checkpoint dictionary.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
    """
    checkpoint_path = checkpoint_dir / f"{prefix}_{model_name}.pt"

    if not checkpoint_path.exists():
        available = list_available_models(checkpoint_dir, prefix)
        raise FileNotFoundError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

    logger.info(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    return checkpoint


def generate_text(model: nn.Module, encoder: TextEncoder, max_tokens: int = 300) -> str:
    """Generate text using the trained model.

    Args:
        model: Trained language model with generate() method.
        encoder: TextEncoder for decoding output.
        max_tokens: Number of tokens to generate.

    Returns:
        Generated text string.
    """
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_indices = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    return encoder.decode(generated_indices)
