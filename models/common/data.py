"""Common data processing utilities for language models."""

import logging
from typing import List, Tuple

import torch

from models.model_config import DEVICE

logger = logging.getLogger(__name__)

#
UNKNOWN_TOKEN = 1000


class TextEncoder:
    """Handles encoding and decoding of text to/from integer sequences."""

    def __init__(self, text: str):
        """Initialize encoder with vocabulary from text.

        Args:
            text: The training text to build vocabulary from.
        """
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # mapping from chars to integers and vice versa
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        logger.info(f"Vocab size: {self.vocab_size}")

    def encode(self, chars: str) -> List[int]:
        """Encode string to list of integers."""
        encode_list = []
        for c in chars:
            num = self.stoi.get(c, UNKNOWN_TOKEN)
            encode_list.append(num)
        return encode_list

    def decode(self, indices: List[int]) -> str:
        """Decode list of integers back to string."""
        index_list = []
        for i in indices:
            index_list.append(self.itos[i])
        return "".join(index_list)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "TextEncoder":
        """Recreate encoder from checkpoint data.

        Args:
            checkpoint: Checkpoint dictionary with vocab_size, stoi, itos.

        Returns:
            TextEncoder instance.
        """
        encoder = cls.__new__(cls)
        encoder.vocab_size = checkpoint["vocab_size"]
        encoder.stoi = checkpoint["stoi"]
        encoder.itos = checkpoint["itos"]
        return encoder


class DataLoader:
    """Handles loading and batching of training data."""

    def __init__(
        self, text: str, encoder: TextEncoder, batch_size: int, block_size: int
    ):
        """Initialize data loader with text and encoder.

        Args:
            text: The training text.
            encoder: TextEncoder instance for encoding.
            batch_size: Number of sequences per batch.
            block_size: Maximum context length.
        """
        self.encoder = encoder
        self.batch_size = batch_size
        self.block_size = block_size
        self.data = torch.tensor(encoder.encode(text), dtype=torch.long)

        # Train test split
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        logger.info(f"Train data shape: {self.train_data.shape}")
        logger.info(f"Validation data shape: {self.val_data.shape}")

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of training or validation data.

        Args:
            split: Either 'train' or 'val'.

        Returns:
            Tuple of (input_batch, target_batch) tensors.
        """
        if split == "train":
            data = self.train_data
        else:
            data = self.val_data

        x_batch = []
        y_batch = []

        start_indices_batch = torch.randint(
            len(data) - self.block_size, (self.batch_size,)
        )  # get start indices for batch

        for idx in start_indices_batch:
            x = data[idx : idx + self.block_size]  # get input sequence
            y = data[
                idx + 1 : idx + self.block_size + 1
            ]  # get target sequence (input shifted by one)
            x_batch.append(x)
            y_batch.append(y)

        return torch.stack(x_batch).to(DEVICE), torch.stack(y_batch).to(DEVICE)
