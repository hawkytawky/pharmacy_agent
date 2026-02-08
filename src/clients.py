"""Client configurations for external services (OpenAI, Supabase)."""

import logging
import os
from typing import List

from openai import OpenAI
from supabase import Client, create_client

from config.chat_config import EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    """
    Creates and returns a Supabase client instance.

    Returns:
        Supabase client configured with environment variables.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY is not set.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in environment variables"
        )

    return create_client(supabase_url, supabase_key)


def create_embedding(text: str) -> List[float]:
    """
    Creates an embedding vector for the given text using OpenAI's embedding model.

    Args:
        text: The text to create an embedding for.

    Returns:
        List of floats representing the embedding vector (1536 dimensions).

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in environment variables")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)

    return response.data[0].embedding
