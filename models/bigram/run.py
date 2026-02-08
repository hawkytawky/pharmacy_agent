"""Main entry point for training the Bigram Language Model.

Usage:
    python -m models.bigram.run --input models/training_data/goethe.txt
    python -m models.bigram.run --input models/training_data/shakespear.txt
"""

import argparse
import logging
from pathlib import Path

import torch

from models.bigram.data import DataLoader, TextEncoder
from models.bigram.generate import generate_text
from models.bigram.model import BigramLanguageModel
from models.bigram.training import train
from models.model_config import DEVICE, TEXT_FILE_PATH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(input_path: str) -> None:
    """Main training function.

    Args:
        input_path: Path to the input text file.
    """
    torch.manual_seed(1337)

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model_name = input_file.stem

    logger.info(f"Loading training data from: {input_path}")
    logger.info(f"Model will be saved as: bigram_{model_name}.pt")

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Text length: {len(text)} characters")
    logger.info(f"Device: {DEVICE}")

    encoder = TextEncoder(text)
    data_loader = DataLoader(text, encoder)

    model = BigramLanguageModel(encoder.vocab_size)
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    train(model, data_loader, encoder, model_name)

    logger.info("\n--- Generated Text Sample ---")
    generated = generate_text(model, encoder, max_tokens=300)
    print(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Bigram Language Model")
    parser.add_argument(
        "--input",
        type=str,
        default=TEXT_FILE_PATH,
        help="Path to the input text file for training",
    )
    args = parser.parse_args()

    main(args.input)
