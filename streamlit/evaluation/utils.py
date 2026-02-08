"""Shared utilities for evaluation pages."""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_results(res_path: Path, pattern: str) -> List[Dict[str, Any]]:
    """Loads all evaluation results matching a pattern from a folder.

    Args:
        res_path: Path to the results directory.
        pattern: Glob pattern for matching result files.

    Returns:
        List of parsed JSON result dictionaries.
    """
    results = []
    if not res_path.exists():
        return results
    for file in res_path.glob(pattern):
        with open(file, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def load_test_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Loads the test dataset JSON file.

    Args:
        file_path: Path to the test dataset JSON file.

    Returns:
        List of test case dictionaries.
    """
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("test_cases", [])
