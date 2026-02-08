"""Evaluator logic for tool routing accuracy."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def evaluate_routing(
    expected_tools: List[str],
    message_history: List[Any],
) -> Dict[str, Any]:
    """Compares actual tool calls in history against expected tool names.

    Args:
        expected_tools: Names of tools that should have been called.
        message_history: The complete message history from the agent state.

    Returns:
        Dictionary containing pass status and detailed diff.
    """
    actual_tools: List[str] = []

    for msg in message_history:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            actual_tools.extend([tc["name"] for tc in msg.tool_calls])

    actual_set = set(actual_tools)
    expected_set = set(expected_tools)
    is_passed = actual_set == expected_set

    logger.debug(f"Routing Eval - Expected: {expected_set} | Actual: {actual_set}")

    return {
        "is_passed": is_passed,
        "actual": list(actual_set),
        "expected": list(expected_set),
        "missing": list(expected_set - actual_set),
        "unexpected": list(actual_set - expected_set),
    }
