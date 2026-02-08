"""Runner for tool routing accuracy evaluation."""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from config.chat_config import LLM_MODEL
from evaluation.tool_routing.evaluator import evaluate_routing
from evaluation.utils import load_test_data, save_results
from src.workflow import PharmacyAssistant

logger = logging.getLogger(__name__)

TEST_DATA_PATH = Path("evaluation/tool_routing/tool_routing_test.json")
OUTPUT_PATH = Path("evaluation/tool_routing/res/")


def run(agent: PharmacyAssistant) -> List[Dict[str, Any]]:
    """Runs the tool routing accuracy evaluation.

    Args:
        agent: The PharmacyAssistant instance.

    Returns:
        List of evaluation result dictionaries.
    """
    test_cases = load_test_data(TEST_DATA_PATH)
    if not test_cases:
        logger.warning("No tool routing test cases found. Skipping.")
        return []

    logger.info(f"Starting TOOL ROUTING evaluation on {len(test_cases)} cases...")

    results = []
    for case in test_cases:
        user_input = case["input"]
        case_id = case["id"]
        expected_tools = case.get("expected_tools", [])

        if not user_input:
            continue

        logger.info(f"Processing Case {case_id}...")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Run agent and collect response
        full_response = ""
        for chunk in agent.stream_chat(user_input, thread_id=thread_id):
            full_response += chunk

        # Get message history from state for tool analysis
        state = agent.graph.get_state(config)
        message_history = state.values.get("messages", [])

        routing_result = evaluate_routing(expected_tools, message_history)

        results.append(
            {
                "test_id": case_id,
                "user_input": user_input,
                "agent_response": full_response,
                "is_passed": routing_result["is_passed"],
                "expected_tools": routing_result["expected"],
                "actual_tools": routing_result["actual"],
                "missing_tools": routing_result["missing"],
                "unexpected_tools": routing_result["unexpected"],
            }
        )

    _save(results)
    return results


def _save(results: List[Dict[str, Any]]):
    """Saves tool routing evaluation results."""
    model_name = LLM_MODEL.replace(":", "-").replace("/", "-")
    output_file = OUTPUT_PATH / f"results_tool_routing_{model_name}.json"

    passed = sum(1 for r in results if r.get("is_passed"))

    meta = {
        "eval_type": "tool_routing",
        "agent_model": LLM_MODEL,
        "total_cases": len(results),
        "passed": passed,
        "pass_rate": f"{(passed / len(results) * 100):.1f}%" if results else "0%",
    }

    save_results(results, output_file, meta)
    logger.info(f"✅ Tool routing evaluation complete! {passed}/{len(results)} passed")
