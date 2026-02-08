"""Training functions for Bigram Language Model."""

import logging
from pathlib import Path
from typing import Dict, List

import torch

from models.bigram.data import DataLoader, TextEncoder
from models.bigram.model import BigramLanguageModel
from models.model_config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    EVAL_INTERVAL,
    EVAL_ITERS,
    LEARNING_RATE,
    MAX_ITERS,
)

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("models/bigram/checkpoints")


@torch.no_grad()
def estimate_loss(model: BigramLanguageModel, data_loader: DataLoader) -> dict:
    """Estimate loss on train and validation sets.

    Args:
        model: The model to evaluate.
        data_loader: DataLoader instance.

    Returns:
        Dictionary with 'train' and 'val' loss estimates.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = data_loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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
    # Create Pytorch optimzer (AdamW more advanced optimzier)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )  # 0.001 learning rate

    logger.info(f"Starting training for {MAX_ITERS} iterations...")

    # Track training history
    history: Dict[str, List] = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
    }

    for steps in range(MAX_ITERS):
        # Evaluate loss at intervals
        if steps % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, data_loader)
            logger.info(
                f"Step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            # Record history
            history["steps"].append(steps)
            history["train_loss"].append(float(losses["train"]))
            history["val_loss"].append(float(losses["val"]))

        # Sample a batch of data
        xb, yb = data_loader.get_batch("train")

        # Forward pass
        logits, loss = model(
            xb, yb
        )  # Calculates the loss for every character in the batch

        # Backward pass and update
        optimizer.zero_grad(set_to_none=True)  # Reset gradients back to zero
        loss.backward()  # Backpropagation to calculate new gradients
        optimizer.step()

    # Final loss
    losses = estimate_loss(model, data_loader)
    logger.info(
        f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    )
    history["steps"].append(MAX_ITERS)
    history["train_loss"].append(float(losses["train"]))
    history["val_loss"].append(float(losses["val"]))

    # Save checkpoint
    save_checkpoint(model, encoder, history, model_name)

    return history


def save_checkpoint(
    model: BigramLanguageModel,
    encoder: TextEncoder,
    history: Dict,
    model_name: str,
) -> Path:
    """Save model checkpoint with encoder and training history.

    Args:
        model: Trained model.
        encoder: TextEncoder instance.
        history: Training history dictionary.
        model_name: Name for the saved model.

    Returns:
        Path to saved checkpoint.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"bigram_{model_name}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": encoder.vocab_size,
        "stoi": encoder.stoi,
        "itos": encoder.itos,
        "training_history": history,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "max_iters": MAX_ITERS,
            "learning_rate": LEARNING_RATE,
            "eval_interval": EVAL_INTERVAL,
            "eval_iters": EVAL_ITERS,
        },
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Model saved to: {checkpoint_path}")

    return checkpoint_path
