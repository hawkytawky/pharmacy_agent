"""Data processing for Bigram Language Model.

Re-exports common utilities with bigram-specific defaults.
"""

from models.common.data import DataLoader as BaseDataLoader
from models.common.data import TextEncoder
from models.model_config import BIGRAM_BATCH_SIZE, BIGRAM_BLOCK_SIZE

__all__ = ["TextEncoder", "DataLoader"]


class DataLoader(BaseDataLoader):
    """DataLoader with Bigram-specific default parameters."""

    def __init__(self, text: str, encoder: TextEncoder):
        """Initialize data loader with Bigram defaults.

        Args:
            text: The training text.
            encoder: TextEncoder instance for encoding.
        """
        super().__init__(
            text=text,
            encoder=encoder,
            batch_size=BIGRAM_BATCH_SIZE,
            block_size=BIGRAM_BLOCK_SIZE,
        )
