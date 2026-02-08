"""Bigram Language Model architecture.

- Only looks at one character/word to predict the next one
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Bigram Language Model for character-level text generation."""

    def __init__(self, vocab_size: int):
        """Initialize the Bigram Language Model.

        Args:
            vocab_size: Size of the vocabulary (number of unique characters).
        """
        super().__init__()
        # Init random weights
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculating the loss between a input and target

        - Can be used for training to calculate the loss
        - Can be used for inference to get the logits with highest probability

        Args:
            idx: Input tensor of shape (B, T) containing token indices.
            targets: Optional target tensor of shape (B, T).

        Returns:
            Tuple of (logits, loss). Loss is None during inference.
        """
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)

        # Inference mode since we dont have any target
        if targets is None:
            loss = None
        # Training mode with target
        else:
            # Putting it into pytorch format
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # Calculating loss

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """inference method to predict the next character in the sequence

        Args:
            idx: Starting context tensor of shape (B, T).
            max_new_tokens: Number of new tokens to generate.

        Returns:
            Generated sequence tensor.
        """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  # Get the logits for the current index
            logits = logits[
                :, -1, :
            ]  # Only take the logits from the last time step (prediction)
            probs = F.softmax(
                logits, dim=-1
            )  # Transform it in probabiliy distribution - by using e(x), we increase variance (makes high values much higher, and small values much smaller) from the logits and keep numbers > 0. After we normalize it for getting the density function
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # Multinomial sampling, weighted sampling where higher probabilities are more likely to get picked, but low probabilities can also get picked (we dont just pick the highest probability as in argmax)
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # Building the sequence by concatenating the new index to the current index
        return idx
