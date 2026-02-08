"""Main entry point for training the GPT Language Model.

Usage:
    python -m models.gpt.run --input models/training_data/goethe.txt
    python -m models.gpt.run --input models/training_data/shakespear.txt
"""

import argparse
import logging
from pathlib import Path

import torch

from models.gpt.data import DataLoader, TextEncoder
from models.gpt.generate import generate_text
from models.gpt.model import GPTLanguageModel
from models.gpt.training import train
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
    # Set random seed for reproducibility
    torch.manual_seed(1337)

    # Load text data
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Extract model name from filename (e.g., goethe.txt -> goethe)
    model_name = input_file.stem

    logger.info(f"Loading training data from: {input_path}")
    logger.info(f"Model will be saved as: gpt_{model_name}.pt")

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Text length: {len(text)} characters")
    logger.info(f"Device: {DEVICE}")

    # Initialize encoder and data loader
    encoder = TextEncoder(text)
    data_loader = DataLoader(text, encoder)

    # Initialize model
    model = GPTLanguageModel(encoder.vocab_size)
    model = model.to(DEVICE)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Train the model (saves checkpoint automatically)
    train(model, data_loader, encoder, model_name)

    # Generate sample text
    logger.info("\n--- Generated Text Sample ---")
    generated = generate_text(model, encoder, max_tokens=300)
    print(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT Language Model")
    parser.add_argument(
        "--input",
        type=str,
        default=TEXT_FILE_PATH,
        help="Path to the input text file for training",
    )
    args = parser.parse_args()

    main(args.input)
