"""LLM page for the Pharmacy Assistant."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_bigram import render_bigram_tab

sys.path.insert(0, str(Path(__file__).parent.parent))
from auth import login_barrier
from components import hide_default_nav, render_sidebar

login_barrier()
st.set_page_config(page_title="LLM", page_icon="🧠", layout="centered")
hide_default_nav()
render_sidebar()

st.title("🧠 LLM")
st.caption("Information about the underlying language models.")

tab_bigram, tab_gpt = st.tabs(["Bigram", "GPT (coming soon)"])

with tab_bigram:
    render_bigram_tab()
    st.divider()

with tab_gpt:
    st.info("🚧 GPT model coming soon...")
    st.markdown(
        """
        The GPT (Generative Pre-trained Transformer) model will be a more advanced
        character-level language model with:
        - **Self-Attention** mechanism
        - **Multiple layers** for deeper understanding
        - **Context awareness** for better text generation
        """
    )
