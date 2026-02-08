"""GPT Language Model architecture.

A character-level Transformer model with self-attention mechanism.
Unlike the Bigram model, GPT can understand context and long-range dependencies.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.model_config import (
    GPT_BLOCK_SIZE,
    GPT_DROPOUT,
    GPT_N_EMBED,
    GPT_N_HEADS,
    GPT_N_LAYER,
)


class Head(nn.Module):
    """One head of self-attention.

    # Attention -> Communication mechansims between embeddings
    # Head of self attention - independece!!!
    # Weighted average
    # Every embedding will omit two vectors:
    # what am I looking for ? -> q (query),
    # what do I contain? -> k (key)
    # Dot producct between q und k -> if dot product is high, attention of those two embeddings is high, otherwise low
    # Value: Q and K found themselves with high dot product, now we say "if they are so important to eachother, which information do I take from it?"
    # K, Q, V are all "abklatsch" from x (transformations)
    """

    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch size, time steps, embedding dimension

        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Perform weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel.

    # we just create multiple heads, since they are independent, we can just run them
    # in parallel and concatenate their outputs together in the end
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embed: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)]
        )
        # we need to project the concatenated output of the heads back to the original embedding dimension
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer Block: communication followed by computation."""

    def __init__(self, n_embed: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections -> we add the input to the output of the attention and feed forward layers,
        # this helps with training stability and allows the model to learn identity functions if needed
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """GPT Language Model for character-level text generation.

    A Transformer-based model that can understand context and generate coherent text.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embed: int = GPT_N_EMBED,
        n_heads: int = GPT_N_HEADS,
        n_layer: int = GPT_N_LAYER,
        block_size: int = GPT_BLOCK_SIZE,
        dropout: float = GPT_DROPOUT,
    ):
        """Initialize the GPT Language Model.

        Args:
            vocab_size: Size of the vocabulary (number of unique characters).
            n_embed: Embedding dimension.
            n_heads: Number of attention heads.
            n_layer: Number of transformer blocks.
            block_size: Maximum context length for predictions.
            dropout: Dropout rate.
        """
        super().__init__()
        self.block_size = block_size

        # Init random weights
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embed
        )  # char -> embedding vector
        self.position_embedding_table = nn.Embedding(
            block_size, n_embed
        )  # position of char -> embedding vector
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layer)],
        )
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)  # embedding vector -> char

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
        B, T = idx.shape
        device = idx.device

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(
            idx
        )  # (batch, time, embedding_dim) -> embedding vector for each character in the input sequence (e.g. "H" -> [0.1, 0.2, 0.3, ...])
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (batch, time, embedding_dim) -> embedding vector for each position in the input sequence from embedding table (e.g. position 0 -> [0.1, 0.2, 0.3, ...]))
        x = (
            tok_emb + pos_emb
        )  # (batch, time, embedding_dim) -> adding the token embedding and position embedding together to get the final embedding for each character in the input sequence (e.g. "H" at position 0 -> [0.2, 0.4, 0.6, ...])
        x = self.blocks(x)  # apply self attention head to the embeddings (B, T, C)
        x = self.ln_f(x)  # apply final layer norm
        logits = self.lm_head(x)  # (batch, time, vocab_size)

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
            # Get the last block_size characters from the input sequence to use as context for prediction
            idx_cond = idx[:, -self.block_size :]
            logits, loss = self(idx_cond)  # Get the logits for the current index
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
