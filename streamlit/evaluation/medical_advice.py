"""Medical Advice KPI rendering for the evaluation page."""

from pathlib import Path

import streamlit as st

from .utils import load_results

MED_ADVICE_RES_PATH = (
    Path(__file__).parent.parent.parent / "evaluation" / "med_advice" / "res"
)


def render():
    """Renders the Medical Advice KPI section."""
    st.markdown("""
    This evaluation tests whether the LLM correctly **refuses to provide medical advice**.
    A judge LLM reviews each response to verify the agent declined appropriately.
    """)

    results = load_results(MED_ADVICE_RES_PATH, "results_medical_advice_*.json")

    if not results:
        st.warning("No evaluation results found.")
        return

    st.subheader("📈 Overall Accuracy")
    results.sort(
        key=lambda x: float(x["meta"]["pass_rate"].replace("%", "")), reverse=True
    )
    judge_model = results[0]["meta"]["judge_model"]
    st.info(f"**Judge Model:** `{judge_model}`")

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
                delta=f"{passed}/{total} passed",
            )

    st.markdown("---")

    _render_detailed_results(results)


def _render_detailed_results(results):
    """Renders the detailed results expander for medical advice evaluation."""
    with st.expander("📋 View detailed results"):
        model_names = [r["meta"]["agent_model"] for r in results]
        selected_model = st.selectbox(
            "Select model", model_names, key="med_advice_model"
        )

        selected_result = next(
            r for r in results if r["meta"]["agent_model"] == selected_model
        )

        for case in selected_result["results"]:
            verdict_icon = "✅" if case["verdict"] == "PASS" else "❌"
            st.markdown(f"**{case['test_id']}** {verdict_icon} - {case['user_input']}")
            if case["verdict"] != "PASS":
                st.error(f"**Response:** {case['agent_response']}")
                st.warning(f"**Reasoning:** {case['reasoning']}")
