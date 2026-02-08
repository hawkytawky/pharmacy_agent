"""Main entry point for running evaluations."""

import logging

from dotenv import load_dotenv

from config.eval_config import EVAL_TYPES
from evaluation.med_advice import runner as med_advice_runner
from evaluation.tool_routing import runner as tool_routing_runner
from src.workflow import PharmacyAssistant

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Runs configured evaluations based on EVAL_TYPES config."""
    logger.info(f"Starting evaluation pipeline for: {EVAL_TYPES}")

    agent = PharmacyAssistant()

    if "medical_advice" in EVAL_TYPES:
        med_advice_runner.run(agent)

    if "tool_routing" in EVAL_TYPES:
        tool_routing_runner.run(agent)

    logger.info("🎉 All evaluations complete!")


if __name__ == "__main__":
    main()
