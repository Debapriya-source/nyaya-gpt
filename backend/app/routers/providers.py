import os

import httpx
from fastapi import APIRouter

from app.models.schemas import ProviderStatusResponse

router = APIRouter()


@router.get("/api/providers/status")
async def provider_status() -> ProviderStatusResponse:
    """Check Ollama availability and list its downloaded models."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_available = False
    ollama_models: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/api/version")
            if resp.status_code == 200:
                ollama_available = True
                tags_resp = await client.get(f"{base_url}/api/tags")
                if tags_resp.status_code == 200:
                    models = tags_resp.json().get("models", [])
                    ollama_models = [m["name"] for m in models]
    except httpx.HTTPError:
        pass

    return ProviderStatusResponse(
        ollama_available=ollama_available,
        ollama_models=ollama_models,
    )
