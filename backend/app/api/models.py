"""Modellarni boshqarish endpointlari."""
import json

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.core.ollama_init import ensure_models

router = APIRouter(prefix="/api/models", tags=["models"])


class PullRequest(BaseModel):
    name: str


@router.get("")
async def list_models():
    """Ollama'da mavjud modellar ro'yxati."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ensure")
async def ensure():
    """Konfiguratsiyadagi modellarni avtomatik yuklash (LLM + Embedding)."""
    await ensure_models()
    return {"ok": True, "llm": settings.LLM_MODEL, "embedding": settings.EMBEDDING_MODEL}


@router.post("/pull")
async def pull(req: PullRequest):
    """Aniq modelni yuklash — streaming progress qaytaradi (SSE)."""

    async def gen():
        try:
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream(
                    "POST",
                    f"{settings.OLLAMA_BASE_URL}/api/pull",
                    json={"name": req.name, "stream": True},
                ) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
