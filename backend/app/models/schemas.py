from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Incoming chat request with user message and LLM provider config."""

    message: str
    provider: Literal["groq", "ollama"]
    ollama_model: str | None = None


class ProviderStatusResponse(BaseModel):
    """Response indicating Ollama availability and its downloaded models."""

    ollama_available: bool
    ollama_models: list[str]
