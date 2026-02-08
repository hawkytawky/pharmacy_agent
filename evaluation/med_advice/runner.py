"""Runner for medical advice guardrail evaluation."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from config.chat_config import LLM_MODEL
from config.eval_config import JUDGE_LLM_MODEL
from evaluation.med_advice.med_evaluator import MedicalSafetyEvaluator
from evaluation.utils import generate_agent_response, load_test_data, save_results
from src.workflow import PharmacyAssistant

logger = logging.getLogger(__name__)

TEST_DATA_PATH = Path("evaluation/med_advice/med_advice_test.json")
OUTPUT_PATH = Path("evaluation/med_advice/res/")


def run(agent: PharmacyAssistant) -> List[Dict[str, Any]]:
    """Runs the medical advice guardrail evaluation.

    Args:
        agent: The PharmacyAssistant instance.

    Returns:
        List of evaluation result dictionaries.
    """
    test_cases = load_test_data(TEST_DATA_PATH)
    if not test_cases:
        logger.warning("No medical advice test cases found. Skipping.")
        return []

    judge = MedicalSafetyEvaluator()
    logger.info(f"Starting MEDICAL ADVICE evaluation on {len(test_cases)} cases...")

    results = []
    for case in test_cases:
        user_input = case.get("input")
        case_id = case.get("id")

        if not user_input:
            continue

        logger.info(f"Processing Case {case_id}...")

        agent_response = generate_agent_response(agent, user_input)
        eval_result = judge.evaluate_compliance(user_input, agent_response)

        results.append(
            {
                "test_id": case_id,
                "user_input": user_input,
                "agent_response": agent_response,
                "verdict": eval_result["verdict"],
                "reasoning": eval_result["reasoning"],
            }
        )

    _save(results)
    return results


def _save(results: List[Dict[str, Any]]):
    """Saves medical advice evaluation results."""
    model_name = LLM_MODEL.replace(":", "-").replace("/", "-")
    output_file = OUTPUT_PATH / f"results_medical_advice_{model_name}.json"

    passed = sum(1 for r in results if r.get("verdict") == "PASS")

    meta = {
        "eval_type": "medical_advice",
        "agent_model": LLM_MODEL,
        "judge_model": JUDGE_LLM_MODEL,
        "total_cases": len(results),
        "passed": passed,
        "pass_rate": f"{(passed / len(results) * 100):.1f}%" if results else "0%",
    }

    save_results(results, output_file, meta)
    logger.info(
        f"✅ Medical advice evaluation complete! {passed}/{len(results)} passed"
    )
