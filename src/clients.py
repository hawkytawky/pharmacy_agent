"""Client configurations for external services (OpenAI, Supabase)."""

import logging
from typing import List

from openai import OpenAI
from supabase import Client, create_client

from src.config.chat_config import EMBEDDING_MODEL
from src.config.secrets import get_secret

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client instance.

    Loads credentials from Streamlit secrets or environment variables.

    Returns:
        Supabase client configured with secrets.

    Raises:
        KeyError: If SUPABASE_URL or SUPABASE_KEY is not set.
    """
    supabase_url = get_secret("SUPABASE_URL")
    supabase_key = get_secret("SUPABASE_KEY")

    return create_client(supabase_url, supabase_key)


def create_embedding(text: str) -> List[float]:
    """
    Creates an embedding vector for the given text using OpenAI's embedding model.

    Loads API key from Streamlit secrets or environment variables.

    Args:
        text: The text to create an embedding for.

    Returns:
        List of floats representing the embedding vector (1536 dimensions).

    Raises:
        KeyError: If OPENAI_API_KEY is not set.
    """
    api_key = get_secret("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)

    return response.data[0].embedding
