"""Training functions for GPT Language Model."""

from pathlib import Path
from typing import Dict

from models.common.training import train_model
from models.gpt.data import DataLoader, TextEncoder
from models.gpt.model import GPTLanguageModel
from models.model_config import (
    GPT_BATCH_SIZE,
    GPT_BLOCK_SIZE,
    GPT_DROPOUT,
    GPT_EVAL_INTERVAL,
    GPT_EVAL_ITERS,
    GPT_LEARNING_RATE,
    GPT_MAX_ITERS,
    GPT_N_EMBED,
    GPT_N_HEADS,
    GPT_N_LAYER,
)

CHECKPOINT_DIR = Path("models/gpt/checkpoints")
MODEL_PREFIX = "gpt"


def train(
    model: GPTLanguageModel,
    data_loader: DataLoader,
    encoder: TextEncoder,
    model_name: str,
) -> Dict:
    """Train the GPT Language Model and save checkpoint.

    Args:
        model: The model to train.
        data_loader: DataLoader instance.
        encoder: TextEncoder instance (needed for saving).
        model_name: Name for the saved model (e.g., 'goethe', 'shakespeare').

    Returns:
        Dictionary with training history.
    """
    hyperparameters = {
        "batch_size": GPT_BATCH_SIZE,
        "block_size": GPT_BLOCK_SIZE,
        "max_iters": GPT_MAX_ITERS,
        "learning_rate": GPT_LEARNING_RATE,
        "eval_interval": GPT_EVAL_INTERVAL,
        "eval_iters": GPT_EVAL_ITERS,
        "n_embed": GPT_N_EMBED,
        "n_heads": GPT_N_HEADS,
        "n_layer": GPT_N_LAYER,
        "dropout": GPT_DROPOUT,
    }

    return train_model(
        model=model,
        data_loader=data_loader,
        encoder=encoder,
        model_name=model_name,
        checkpoint_dir=CHECKPOINT_DIR,
        prefix=MODEL_PREFIX,
        max_iters=GPT_MAX_ITERS,
        eval_interval=GPT_EVAL_INTERVAL,
        eval_iters=GPT_EVAL_ITERS,
        learning_rate=GPT_LEARNING_RATE,
        hyperparameters=hyperparameters,
    )
