"""Utilities for loading Streamlit secrets in the Streamlit environment."""

import os

import streamlit as st


def get_secret(key: str, default: str = None) -> str:
    """
    Get a secret value from Streamlit secrets or environment variables.

    In Streamlit context, tries to load from st.secrets first (production and local dev).
    Falls back to environment variables for non-Streamlit contexts (e.g., scripts, evaluations).

    Args:
        key: The secret key to retrieve.
        default: Default value if key is not found.

    Returns:
        The secret value or default.

    Raises:
        KeyError: If key not found and no default provided.
    """
    try:
        # Try Streamlit secrets first (works in both local dev and Streamlit Cloud)
        return st.secrets[key]
    except (KeyError, AttributeError):
        # Fall back to environment variables for non-Streamlit contexts
        value = os.getenv(key, default)
        if value is None:
            raise KeyError(
                f"Secret '{key}' not found in st.secrets or environment variables"
            )
        return value
