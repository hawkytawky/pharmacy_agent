"""Database explorer page for viewing medication data samples."""

from auth import login_barrier
from components import hide_default_nav, render_sidebar

import streamlit as st
from src.clients import get_supabase_client

login_barrier()
hide_default_nav()
render_sidebar()

st.title("🗄️ Database Explorer")

supabase = get_supabase_client()

st.caption("Browse medication data from the database")
st.divider()


def fetch_all_records(table_name: str = "medications") -> list:
    """Fetches all records from the specified database table.

    Args:
        table_name: Name of the Supabase table to query.

    Returns:
        List of records.
    """
    try:
        response = supabase.table(table_name).select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return []


def get_random_medication(records: list) -> dict:
    """Returns a random medication record.

    Args:
        records: List of medication records.

    Returns:
        Random medication record.
    """
    if not records:
        return None
    import random

    return random.choice(records)


# Fetch data
records = fetch_all_records("medications")

if records:
    # Total records metric
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("📊 Total Records", len(records))
    with col2:
        if st.button(
            "🎲 Get Random Medication", type="primary", use_container_width=True
        ):
            st.session_state.current_medication = get_random_medication(records)

    st.divider()

    # Display current medication
    if "current_medication" in st.session_state:
        med = st.session_state.current_medication

        if med:
            # Name as header
            st.subheader(f"💊 {med.get('name', 'N/A')}")

            st.divider()

            # Display JSON fields
            json_fields = ["base_info", "stock", "regulatory", "medical_data"]

            for field in json_fields:
                if field in med:
                    st.write(f"**{field}:**")
                    st.json(med[field])
                    st.divider()

else:
    st.warning("No data found in the medications table.")
