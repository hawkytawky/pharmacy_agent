"""Data processing for GPT Language Model.

Re-exports common utilities with GPT-specific defaults.
"""

from models.common.data import DataLoader as BaseDataLoader
from models.common.data import TextEncoder
from models.model_config import GPT_BATCH_SIZE, GPT_BLOCK_SIZE

__all__ = ["TextEncoder", "DataLoader"]


class DataLoader(BaseDataLoader):
    """DataLoader with GPT-specific default parameters."""

    def __init__(self, text: str, encoder: TextEncoder):
        """Initialize data loader with GPT defaults.

        Args:
            text: The training text.
            encoder: TextEncoder instance for encoding.
        """
        super().__init__(
            text=text,
            encoder=encoder,
            batch_size=GPT_BATCH_SIZE,
            block_size=GPT_BLOCK_SIZE,
        )
