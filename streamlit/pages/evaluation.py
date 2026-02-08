"""Evaluation page for the Pharmacy Assistant."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import hide_default_nav, render_sidebar

from evaluation import medical_advice, tool_routing

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")
hide_default_nav()
render_sidebar()

st.title("📊 Evaluation")
st.caption("Evaluation metrics and test results for the Pharmacy Assistant.")
st.warning("""
⚠️ **MVP Disclaimer:** This evaluation is a first draft for demonstration purposes.
Production-ready benchmarks would require more test cases, diverse model comparisons,
and additional KPIs (e.g., costs, edge-case coverage or considering multi-turn interactions).
These metrics provide a surface-level view of how one could approach agent evaluation.
""")

tab_med, tab_tools = st.tabs(["🛡️ Not Giving Medical Advice", "🔧 Tool Routing"])

with tab_med:
    medical_advice.render()

with tab_tools:
    tool_routing.render()
