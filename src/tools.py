"""Tools for the Pharmacy Assistant including medication search functionality."""

import logging
from typing import Dict, List

from langchain_core.tools import tool

from src.clients import create_embedding, get_supabase_client
from src.config.chat_config import RETRIEVAL_MATCH_COUNT, RETRIEVAL_MATCH_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_medication(query: str) -> List[Dict]:
    """
    Searches for medications in the database based on the query string.

    This function creates an embedding of the search query using OpenAI's text-embedding-3-small
    model, then calls the Supabase RPC function match_medications to find similar medications.

    Args:
        query: The search term or medication name to search for.

    Returns:
        List of dictionaries containing medication data with keys:
        id, name, base_info, stock, regulatory, medical_data.
        Returns empty list if no matches found or on error.
    """
    try:
        supabase_client = get_supabase_client()

        # Step 1: Create embedding for the search query
        logger.info(f"Creating embedding for query: {query}")
        query_embedding = create_embedding(query)

        # Step 2: Call Supabase RPC function to find matching medications
        logger.info("Calling Supabase RPC function match_medications")
        response = supabase_client.rpc(
            "match_medications",
            {
                "query_embedding": query_embedding,
                "match_threshold": RETRIEVAL_MATCH_THRESHOLD,
                "match_count": RETRIEVAL_MATCH_COUNT,
            },
        ).execute()

        # Step 3: Process and return results
        if response.data:
            logger.info(f"Found {len(response.data)} matching medications")
            return response.data

        logger.info("No matching medications found")
        return []

    except TimeoutError as e:
        logger.error(f"Timeout error during medication search: {e}")
        return []
    except ConnectionError as e:
        logger.error(f"Network error during medication search: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during medication search: {e}")
        return []


@tool
def get_med_base_info(query: str) -> str:
    """
    Get basic identification and product information for a medication.

    Use this tool when a customer asks about:
    - What form a medication comes in (tablets, capsules, etc.)
    - The brand or manufacturer
    - Price information
    - Pack size

    Args:
        query: The medication name to look up.

    Returns:
        Formatted string with base product information.
    """
    results = search_medication(query)

    if not results:
        return f"No medications found matching '{query}'."

    formatted_results = []
    for med in results:
        base_info = med.get("base_info", {})
        price = base_info.get("price", {})
        med_info = f"""
            **{med.get("name", "Unknown")}**
            - Form: {base_info.get("form", "N/A")}
            - Brand: {base_info.get("brand", "N/A")}
            - Pack Size: {base_info.get("pack_size", "N/A")}
            - Price: {price.get("amount", "N/A")} {price.get("currency", "")}
        """.strip()
        formatted_results.append(med_info)

    return "\n---\n".join(formatted_results)


@tool
def get_med_stock_info(query: str) -> str:
    """
    Get stock availability and location information for a medication.

    Use this tool when a customer asks about:
    - Whether a medication is in stock
    - How many units are available
    - Where to find it in the store

    Args:
        query: The medication name to look up.

    Returns:
        Formatted string with stock and availability information.
    """
    results = search_medication(query)

    if not results:
        return f"No medications found matching '{query}'."

    formatted_results = []
    for med in results:
        stock = med.get("stock", {})
        med_info = f"""
            **{med.get("name", "Unknown")}**
            - Status: {stock.get("status", "N/A")}
            - Quantity: {stock.get("quantity", "N/A")} units
            - Location: {stock.get("location", "N/A")}
        """.strip()
        formatted_results.append(med_info)

    return "\n---\n".join(formatted_results)


@tool
def get_med_regulatory_info(query: str) -> str:
    """
    Get regulatory and legal information for a medication.

    Use this tool when a customer asks about:
    - Whether a prescription is required
    - Age restrictions for purchase
    - Whether it's a controlled/narcotic substance

    Args:
        query: The medication name to look up.

    Returns:
        Formatted string with regulatory information.
    """
    results = search_medication(query)

    if not results:
        return f"No medications found matching '{query}'."

    formatted_results = []
    for med in results:
        regulatory = med.get("regulatory", {})
        med_info = f"""
            **{med.get("name", "Unknown")}**
            - Requires Prescription: {"Yes" if regulatory.get("requires_prescription") else "No"}
            - Age Restriction: {regulatory.get("age_restriction", "None")} years
            - Narcotic: {"Yes" if regulatory.get("narcotic") else "No"}
        """.strip()
        formatted_results.append(med_info)

    return "\n---\n".join(formatted_results)


@tool
def get_med_safety_info(query: str) -> str:
    """
    Get medical and safety information from the package insert.

    Use this tool when a customer asks about:
    - Active ingredients
    - Application areas / indications
    - Dosage and administration instructions
    - Side effects or contraindications (only read from package insert, no medical advice)

    Args:
        query: The medication name to look up.

    Returns:
        Formatted string with medical data from the package insert.
    """
    results = search_medication(query)

    if not results:
        return f"No medications found matching '{query}'."

    formatted_results = []
    for med in results:
        medical = med.get("medical_data", {})
        safety = medical.get("safety_warnings", {})
        usage = medical.get("usage_instructions", {})

        contraindications = safety.get("contraindications", [])
        contraindications_str = (
            ", ".join(contraindications) if contraindications else "N/A"
        )

        ingredients = medical.get("active_ingredients", [])
        ingredients_str = ", ".join(ingredients) if ingredients else "N/A"

        areas = medical.get("application_area", [])
        areas_str = ", ".join(areas) if areas else "N/A"

        med_info = f"""
            **{med.get("name", "Unknown")}**
            - Active Ingredients: {ingredients_str}
            - Application Area: {areas_str}
            - Dosage: {usage.get("dosage", "N/A")}
            - Administration: {usage.get("administration", "N/A")}
            - Side Effects: {safety.get("side_effects_summary", "N/A")}
            - Contraindications: {contraindications_str}
        """.strip()
        formatted_results.append(med_info)

    return "\n---\n".join(formatted_results)


# List of all available tools for the agent
pharmacy_tools = [
    get_med_base_info,
    get_med_stock_info,
    get_med_regulatory_info,
    get_med_safety_info,
]
