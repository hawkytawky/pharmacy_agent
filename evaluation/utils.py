"""Shared utility functions for evaluation modules."""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from src.workflow import PharmacyAssistant

logger = logging.getLogger(__name__)


def load_test_data(file_path: Path) -> List[Dict[str, Any]]:
    """Loads test cases from a JSON file.

    Args:
        file_path: Path to the input JSON file.

    Returns:
        A list of test case dictionaries.
    """
    try:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("test_cases", [])

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        return []


def save_results(
    results: List[Dict[str, Any]],
    output_path: Path,
    meta: Dict[str, Any],
):
    """Saves evaluation results to a JSON file.

    Args:
        results: The evaluation results.
        output_path: Path to save the results.
        meta: Metadata dictionary to include in output.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            final_output = {"meta": meta, "results": results}
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def generate_agent_response(agent: PharmacyAssistant, user_input: str) -> str:
    """Runs the user input through the PharmacyAssistant graph.

    Generates a NEW thread_id for every call to ensure statelessness.

    Args:
        agent: The PharmacyAssistant instance.
        user_input: The user query to process.

    Returns:
        The agent's full response as a string.
    """
    thread_id = str(uuid.uuid4())
    full_response = ""

    try:
        for chunk in agent.stream_chat(user_input, thread_id=thread_id):
            full_response += chunk
        return full_response

    except Exception as e:
        logger.error(f"Agent generation failed: {e}")
        return f"[ERROR generating response: {str(e)}]"
