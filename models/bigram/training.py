"""Training functions for Bigram Language Model."""

from pathlib import Path
from typing import Dict

from models.bigram.data import DataLoader, TextEncoder
from models.bigram.model import BigramLanguageModel
from models.common.training import train_model
from models.model_config import (
    BIGRAM_BATCH_SIZE,
    BIGRAM_BLOCK_SIZE,
    BIGRAM_EVAL_INTERVAL,
    BIGRAM_EVAL_ITERS,
    BIGRAM_LEARNING_RATE,
    BIGRAM_MAX_ITERS,
)

CHECKPOINT_DIR = Path("models/bigram/checkpoints")
MODEL_PREFIX = "bigram"


def train(
    model: BigramLanguageModel,
    data_loader: DataLoader,
    encoder: TextEncoder,
    model_name: str,
) -> Dict:
    """Train the Bigram Language Model and save checkpoint.

    Args:
        model: The model to train.
        data_loader: DataLoader instance.
        encoder: TextEncoder instance (needed for saving).
        model_name: Name for the saved model (e.g., 'goethe', 'shakespeare').

    Returns:
        Dictionary with training history.
    """
    hyperparameters = {
        "batch_size": BIGRAM_BATCH_SIZE,
        "block_size": BIGRAM_BLOCK_SIZE,
        "max_iters": BIGRAM_MAX_ITERS,
        "learning_rate": BIGRAM_LEARNING_RATE,
        "eval_interval": BIGRAM_EVAL_INTERVAL,
        "eval_iters": BIGRAM_EVAL_ITERS,
    }

    return train_model(
        model=model,
        data_loader=data_loader,
        encoder=encoder,
        model_name=model_name,
        checkpoint_dir=CHECKPOINT_DIR,
        prefix=MODEL_PREFIX,
        max_iters=BIGRAM_MAX_ITERS,
        eval_interval=BIGRAM_EVAL_INTERVAL,
        eval_iters=BIGRAM_EVAL_ITERS,
        learning_rate=BIGRAM_LEARNING_RATE,
        hyperparameters=hyperparameters,
    )
