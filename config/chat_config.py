"""Configuration parameters for the Pharmacy Assistant."""

from typing import Literal

# LLM Provider: "openai" or "ollama"
LLM_PROVIDER: Literal["openai", "ollama"] = "openai"

# OpenAI Configuration
OPENAI_MODEL = "gpt-5-mini-2025-08-07"
OPENAI_TEMPERATURE = 0.0

# Ollama Configuration
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TEMPERATURE = 0.0

# Active LLM settings (based on provider)
LLM_MODEL = OPENAI_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL
LLM_TEMPERATURE = OPENAI_TEMPERATURE if LLM_PROVIDER == "openai" else OLLAMA_TEMPERATURE

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Retrieval Configuration
RETRIEVAL_MATCH_THRESHOLD = 0.5
RETRIEVAL_MATCH_COUNT = 3
