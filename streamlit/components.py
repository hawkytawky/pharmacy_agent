"""Shared UI components for the Streamlit app."""

import uuid

from auth import logout

import streamlit as st


def hide_default_nav():
    """Hide the default Streamlit page navigation and reduce top padding."""
    st.markdown(
        """
        <style>
        [data-testid='stSidebarNav'] {display: none;}
        .block-container {padding-top: 2rem; padding-bottom: 1rem;}
        header[data-testid='stHeader'] {height: 0rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(show_clear_chat: bool = False):
    """Render the sidebar with navigation and controls.

    Args:
        show_clear_chat: Whether to show the Clear Chat button.
    """
    with st.sidebar:
        st.page_link("app.py", label="Chat", icon="💬")
        st.page_link("pages/evaluation.py", label="Evaluation", icon="📊")
        st.page_link("pages/docs.py", label="Documentation", icon="📄")
        st.page_link("pages/llm.py", label="LLM", icon="🧠")
        st.page_link("pages/database.py", label="Database", icon="🗄️")

        if show_clear_chat:
            st.divider()
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()
