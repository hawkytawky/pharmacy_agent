"""LLM page for the Pharmacy Assistant."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_bigram import render_bigram_tab
from llm_gpt import render_gpt_tab

sys.path.insert(0, str(Path(__file__).parent.parent))
from auth import login_barrier
from components import hide_default_nav, render_sidebar

login_barrier()
st.set_page_config(page_title="GPT", page_icon="🧠", layout="centered")
hide_default_nav()
render_sidebar()

st.title("🧠 GPT")
st.info("""ℹ️ The model architectures and training implementation are based on the tutorial
**'Let's build GPT: from scratch, in code, spelled out.'** by the great **Andrej Karpathy**.

[Watch the full lecture by Andrej Karpathy](https://youtu.be/kCc8FmEb1nY)

Note: These are **decoder-only** models. They generate text by continuing from a given context,
but they don't "respond" to input like a chat model. They are trained to predict the next character based on the previous ones.
""")

tab_bigram, tab_gpt = st.tabs(["Bigram", "GPT"])

with tab_bigram:
    render_bigram_tab()
    st.divider()

with tab_gpt:
    render_gpt_tab()
    st.divider()
