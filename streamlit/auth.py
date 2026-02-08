"""Authentication utilities for the Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_password() -> bool:
    """Gibt True zurück, wenn der Benutzer eingeloggt ist, sonst False."""

    def password_entered():
        """Prüft das Passwort gegen die Secrets."""
        if st.session_state["password_input"] == st.secrets["ADMIN_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password_input"]  # Sicherheit: Eingabe löschen
        else:
            st.session_state["password_correct"] = False

    # Falls bereits erfolgreich eingeloggt
    if st.session_state.get("password_correct", False):
        return True

    # Login-Formular anzeigen
    st.title("🔒 Login")
    st.text_input(
        "Bitte Passwort für den Zugriff eingeben:",
        type="password",
        on_change=password_entered,
        key="password_input",
    )

    if (
        "password_correct" in st.session_state
        and not st.session_state["password_correct"]
    ):
        st.error("😕 Passwort falsch")

    return False


def login_barrier():
    """Stoppt die Ausführung der Seite, wenn der User nicht eingeloggt ist."""
    if not check_password():
        st.stop()


def logout():
    """Logs out the current user by clearing the authenticated session state."""
    st.session_state.authenticated = False
    st.rerun()
