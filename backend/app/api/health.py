"""Sogʻlomlik va sozlamalarni tekshirish endpointi."""
import httpx
from fastapi import APIRouter

from app.config import settings
from app.core.vector_store import vector_store

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health():
    status = {"ok": True, "services": {}}

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            status["services"]["ollama"] = {"ok": True, "models": models}
    except Exception as e:
        status["ok"] = False
        status["services"]["ollama"] = {"ok": False, "error": str(e)}

    # Qdrant
    try:
        cols = await vector_store.client.get_collections()
        status["services"]["qdrant"] = {
            "ok": True,
            "collections": [c.name for c in cols.collections],
        }
    except Exception as e:
        status["ok"] = False
        status["services"]["qdrant"] = {"ok": False, "error": str(e)}

    return status
