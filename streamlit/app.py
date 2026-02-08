"""Streamlit chat interface for the Pharmacy Assistant."""

import itertools
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

import streamlit as st

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from auth import check_password
from components import hide_default_nav, render_sidebar

from src.workflow import PharmacyAssistant

load_dotenv(project_root / ".env")


def slow_stream(iterator):
    for chunk in iterator:
        time.sleep(0.02)
        yield chunk


def init_session_state():
    """Initialize session state variables."""
    if "assistant" not in st.session_state:
        st.session_state.assistant = PharmacyAssistant()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


def display_chat_history():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        role = message["role"]
        avatar = "👤" if role == "user" else "💊"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])


def main():
    """Main function to run the Streamlit app."""
    check_password()

    st.set_page_config(
        page_title="Pharmacy Assistant", page_icon="💊", layout="centered"
    )

    hide_default_nav()

    st.title("💊 Pharmacy Assistant")
    st.caption("""Ask me about medication:
               🔍 Basics · 📦 Stock · 📋 Prescription · ⚠️ Safety)""")
    st.caption("""**Note:** This bot will NOT provide medical advice.""")

    init_session_state()
    render_sidebar()
    display_chat_history()

    if user_input := st.chat_input("Ask about a medication..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="💊"):
            stream = st.session_state.assistant.stream_chat(
                user_input, thread_id=st.session_state.thread_id
            )

            with st.spinner("Thinking..."):
                first_chunk = next(stream, "")

            full_generator = itertools.chain([first_chunk], stream)
            full_response = st.write_stream(slow_stream(full_generator))

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
