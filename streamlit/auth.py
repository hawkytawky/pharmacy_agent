"""Authentication utilities for the Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.secrets import get_secret


def check_password() -> bool:
    """
    Checks if the user has entered the correct password.

    Uses Streamlit session state to track authentication status.
    Once authenticated, the user stays logged in for the session.
    Loads the correct password from st.secrets or environment variables.

    Returns:
        True if user is authenticated, False otherwise.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔐 Pharmacy Assistant")
    st.caption("Please enter the password to access the application.")

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password...",
    )

    if st.button("Login", type="primary"):
        correct_password = get_secret("ADMIN_PASSWORD")
        if password == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")

    st.stop()


def logout():
    """Logs out the current user by clearing the authenticated session state."""
    st.session_state.authenticated = False
    st.rerun()
