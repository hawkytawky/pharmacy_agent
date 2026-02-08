"""Common training utilities for language models."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from models.common.data import DataLoader, TextEncoder

logger = logging.getLogger(__name__)


@torch.no_grad()
def estimate_loss(
    model: nn.Module, data_loader: DataLoader, eval_iters: int
) -> Dict[str, float]:
    """Estimate loss on train and validation sets.

    Args:
        model: The model to evaluate.
        data_loader: DataLoader instance.
        eval_iters: Number of iterations for loss estimation.

    Returns:
        Dictionary with 'train' and 'val' loss estimates.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    encoder: TextEncoder,
    model_name: str,
    checkpoint_dir: Path,
    prefix: str,
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    learning_rate: float,
    hyperparameters: Dict[str, Any],
) -> Dict:
    """Train a language model and save checkpoint.

    Args:
        model: The model to train.
        data_loader: DataLoader instance.
        encoder: TextEncoder instance (needed for saving).
        model_name: Name for the saved model (e.g., 'goethe', 'shakespeare').
        checkpoint_dir: Directory to save checkpoints.
        prefix: Model prefix for checkpoint filename (e.g., 'bigram', 'gpt').
        max_iters: Maximum training iterations.
        eval_interval: Steps between loss evaluations.
        eval_iters: Iterations for loss estimation.
        learning_rate: Learning rate for optimizer.
        hyperparameters: Dictionary of hyperparameters to save.

    Returns:
        Dictionary with training history.
    """
    # Create Pytorch optimzer (AdamW more advanced optimzier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    logger.info(f"Starting training for {max_iters} iterations...")

    # Track training history
    history: Dict[str, List] = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
    }

    for steps in range(max_iters):
        # Evaluate loss at intervals
        if steps % eval_interval == 0:
            losses = estimate_loss(model, data_loader, eval_iters)
            logger.info(
                f"Step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            # Record history
            history["steps"].append(steps)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])

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
    losses = estimate_loss(model, data_loader, eval_iters)
    logger.info(
        f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    )
    history["steps"].append(max_iters)
    history["train_loss"].append(losses["train"])
    history["val_loss"].append(losses["val"])

    # Save checkpoint
    save_checkpoint(
        model=model,
        encoder=encoder,
        history=history,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        prefix=prefix,
        hyperparameters=hyperparameters,
    )

    return history


def save_checkpoint(
    model: nn.Module,
    encoder: TextEncoder,
    history: Dict,
    model_name: str,
    checkpoint_dir: Path,
    prefix: str,
    hyperparameters: Dict[str, Any],
) -> Path:
    """Save model checkpoint with encoder and training history.

    Args:
        model: Trained model.
        encoder: TextEncoder instance.
        history: Training history dictionary.
        model_name: Name for the saved model.
        checkpoint_dir: Directory to save checkpoint.
        prefix: Model prefix for filename (e.g., 'bigram', 'gpt').
        hyperparameters: Dictionary of hyperparameters.

    Returns:
        Path to saved checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{prefix}_{model_name}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": encoder.vocab_size,
        "stoi": encoder.stoi,
        "itos": encoder.itos,
        "training_history": history,
        "hyperparameters": hyperparameters,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Model saved to: {checkpoint_path}")

    return checkpoint_path
