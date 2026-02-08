"""LLM page for the Pharmacy Assistant."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import hide_default_nav, render_sidebar

st.set_page_config(page_title="LLM", page_icon="🧠", layout="centered")
hide_default_nav()
render_sidebar()

st.title("🧠 LLM")
st.caption("Information about the underlying language model.")

# Placeholder content
st.info("Coming soon: LLM configuration and details.")
