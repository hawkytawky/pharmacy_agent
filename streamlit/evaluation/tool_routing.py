"""Tool Routing KPI rendering for the evaluation page."""

from pathlib import Path
from typing import Any, Dict, List

import plotly.express as px

import streamlit as st

from .utils import load_results, load_test_dataset

TOOL_ROUTING_RES_PATH = (
    Path(__file__).parent.parent.parent / "evaluation" / "tool_routing" / "res"
)
TOOL_ROUTING_TEST_PATH = (
    Path(__file__).parent.parent.parent
    / "evaluation"
    / "tool_routing"
    / "tool_routing_test.json"
)


def render():
    """Renders the Tool Routing Accuracy KPI section."""
    st.markdown("""
    This evaluation tests whether the agent calls the **correct tools** for each query.
    We measure exact match accuracy, precision, recall, and F1 score across different tool combinations.
    """)

    results = load_results(TOOL_ROUTING_RES_PATH, "results_tool_routing_*.json")
    test_cases = load_test_dataset(TOOL_ROUTING_TEST_PATH)

    if not results:
        st.warning("No evaluation results found.")
        return

    results.sort(
        key=lambda x: float(x["meta"]["pass_rate"].replace("%", "")), reverse=True
    )

    _render_overall_accuracy(results)
    st.markdown("---")

    _render_precision_recall_f1(results)
    st.markdown("---")

    _render_accuracy_by_tool_count(results, test_cases)
    st.markdown("---")

    _render_accuracy_per_tool(results)
    st.markdown("---")

    _render_detailed_results(results)


def _render_overall_accuracy(results: List[Dict[str, Any]]):
    """Renders the overall accuracy metrics section."""
    st.subheader("📈 Overall Accuracy")
    cols = st.columns(len(results))
    for idx, result in enumerate(results):
        meta = result["meta"]
        model_name = meta["agent_model"]
        pass_rate = meta["pass_rate"]
        passed = meta["passed"]
        total = meta["total_cases"]

        with cols[idx]:
            st.metric(
                label=f"**{model_name}**",
                value=pass_rate,
                delta=f"{passed}/{total} exact match",
            )


def _render_precision_recall_f1(results: List[Dict[str, Any]]):
    """Renders the Precision, Recall & F1 section with chart."""
    st.subheader("🎯 Precision, Recall & F1")
    st.info("""
    **What do these numbers mean for our pharmacy agent?**

    **Precision**: *"No unnecessary actions?"* — User asks for *price* -> Agent calls only `base_info`, not `safety_info` too

    **Recall**: *"Nothing forgotten?"* — User asks for *price AND stock* -> Agent calls both tools, not just one

    **F1**: *"Overall reliability?"* — Harmonic mean balancing both error types
    """)

    metrics = _calculate_tool_metrics(results)

    prf_chart_data = []
    for model_name, m in metrics.items():
        prf_chart_data.append(
            {
                "Model": model_name,
                "Metric": "Precision",
                "Score (%)": m["precision"] * 100,
            }
        )
        prf_chart_data.append(
            {"Model": model_name, "Metric": "Recall", "Score (%)": m["recall"] * 100}
        )
        prf_chart_data.append(
            {"Model": model_name, "Metric": "F1", "Score (%)": m["f1"] * 100}
        )

    fig_prf = px.bar(
        prf_chart_data,
        x="Model",
        y="Score (%)",
        color="Metric",
        barmode="group",
        text_auto=".1f",
        color_discrete_map={
            "Precision": "#636EFA",
            "Recall": "#EF553B",
            "F1": "#00CC96",
        },
    )
    fig_prf.update_layout(
        yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_prf, use_container_width=True)


def _render_accuracy_by_tool_count(
    results: List[Dict[str, Any]], test_cases: List[Dict]
):
    """Renders accuracy by number of tools chart."""
    st.subheader("📊 Accuracy by Number of Tools")
    st.write("""
    The evaluation dataset uses a weighted distribution of 40 test cases ($n=16, 12, 8, 4$) to prioritize high-frequency,
    single-intent queries common in real pharmacy settings, while providing more complex multi-tool scenarios to
    stress-test the agent's parallel reasoning limits. This structure enables a granular KPI analysis of how routing
    precision scales with query complexity.
    """)

    if not test_cases:
        return

    first_by_count = _calculate_accuracy_by_tool_count(results[0], test_cases)
    tool_count_n = {k: v["total"] for k, v in first_by_count.items()}

    chart_data = []
    for result in results:
        model_name = result["meta"]["agent_model"]
        by_count = _calculate_accuracy_by_tool_count(result, test_cases)

        for num_tools, stats in by_count.items():
            accuracy = (
                stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            n_cases = tool_count_n.get(num_tools, stats["total"])
            chart_data.append(
                {
                    "Tools": f"{num_tools} Tool(s)\n(n={n_cases})",
                    "Accuracy (%)": accuracy,
                    "Model": model_name,
                    "Cases": stats["total"],
                }
            )

    fig_tools_count = px.bar(
        chart_data,
        x="Tools",
        y="Accuracy (%)",
        color="Model",
        barmode="group",
        text_auto=".1f",
        hover_data=["Cases"],
    )
    fig_tools_count.update_layout(
        yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_tools_count, use_container_width=True)


def _render_accuracy_per_tool(results: List[Dict[str, Any]]):
    """Renders accuracy per individual tool chart."""
    st.subheader("🛠️ Accuracy per Tool")
    st.write("""
    To ensure a balanced evaluation of each tools performance, the dataset follows a uniform distribution where
    each specific tool is called exactly $n=20$ times, preventing statistical bias and ensuring that the final
    accuracy metric reflects the reliable performance of the entire toolset.
    """)

    if not results:
        return

    tool_stats = _calculate_accuracy_by_tool(results)

    first_model = results[0]["meta"]["agent_model"]
    tool_n = {
        tool: stats.get(first_model, {}).get("total", 0)
        for tool, stats in tool_stats.items()
    }

    chart_data = []
    for tool, model_stats in tool_stats.items():
        short_tool = tool.replace("get_med_", "").replace("_info", "")
        n_cases = tool_n.get(tool, 0)
        for model_name, stats in model_stats.items():
            accuracy = (
                stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            chart_data.append(
                {
                    "Tool": f"{short_tool}\n(n={n_cases})",
                    "Accuracy (%)": accuracy,
                    "Model": model_name,
                    "Cases": stats["total"],
                }
            )

    fig_tools = px.bar(
        chart_data,
        x="Tool",
        y="Accuracy (%)",
        color="Model",
        barmode="group",
        text_auto=".1f",
        hover_data=["Cases"],
    )
    fig_tools.update_layout(
        yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_tools, use_container_width=True)


def _render_detailed_results(results: List[Dict[str, Any]]):
    """Renders the detailed results expander."""
    with st.expander("📋 View detailed results"):
        model_names = [r["meta"]["agent_model"] for r in results]
        selected_model = st.selectbox(
            "Select model", model_names, key="tool_routing_model"
        )

        selected_result = next(
            r for r in results if r["meta"]["agent_model"] == selected_model
        )

        for case in selected_result["results"]:
            status_icon = "✅" if case["is_passed"] else "❌"
            st.markdown(f"**{case['test_id']}** {status_icon} - {case['user_input']}")

            if not case["is_passed"]:
                st.error(
                    f"Expected: `{case['expected_tools']}` → Actual: `{case['actual_tools']}`"
                )
                if case["missing_tools"]:
                    st.warning(f"Missing: `{case['missing_tools']}`")
                if case["unexpected_tools"]:
                    st.info(f"Unexpected: `{case['unexpected_tools']}`")


def _calculate_tool_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Calculates Precision, Recall, F1 for each model's tool routing."""
    metrics = {}

    for result in results:
        model_name = result["meta"]["agent_model"]
        tp, fp, fn = 0, 0, 0

        for case in result["results"]:
            expected = set(case["expected_tools"])
            actual = set(case["actual_tools"])

            tp += len(expected & actual)
            fp += len(actual - expected)
            fn += len(expected - actual)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics[model_name] = {"precision": precision, "recall": recall, "f1": f1}

    return metrics


def _calculate_accuracy_by_tool_count(
    results: Dict[str, Any], test_cases: List[Dict]
) -> Dict[int, Dict[str, int]]:
    """Calculates accuracy grouped by number of expected tools."""
    test_case_map = {
        tc["id"]: tc.get("no_tools", len(tc.get("expected_tools", [])))
        for tc in test_cases
    }

    by_count = {}
    for case in results["results"]:
        no_tools = test_case_map.get(case["test_id"], 1)
        if no_tools not in by_count:
            by_count[no_tools] = {"passed": 0, "total": 0}
        by_count[no_tools]["total"] += 1
        if case["is_passed"]:
            by_count[no_tools]["passed"] += 1

    return by_count


def _calculate_accuracy_by_tool(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Calculates accuracy per individual tool across all models."""
    tool_stats = {}

    for result in results:
        model_name = result["meta"]["agent_model"]

        for case in result["results"]:
            expected = set(case["expected_tools"])
            actual = set(case["actual_tools"])

            for tool in expected:
                if tool not in tool_stats:
                    tool_stats[tool] = {}
                if model_name not in tool_stats[tool]:
                    tool_stats[tool][model_name] = {"correct": 0, "total": 0}

                tool_stats[tool][model_name]["total"] += 1
                if tool in actual:
                    tool_stats[tool][model_name]["correct"] += 1

    return tool_stats
