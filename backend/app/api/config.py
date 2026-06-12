"""Runtime config endpointlari — UI'dan model va sozlamalarni o'zgartirish."""
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.core.embeddings import embedder
from app.core.runtime_config import EDITABLE_FIELDS, runtime_config
from app.core.vector_store import vector_store

router = APIRouter(prefix="/api/config", tags=["config"])


class UpdateRequest(BaseModel):
    # Erkin shaklda — backend tomonda validatsiya qilamiz
    updates: dict[str, Any]


@router.get("")
async def get_config():
    """Joriy sozlamalar + Ollama'da mavjud modellar (UI dropdown uchun)."""
    available: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                for m in r.json().get("models", []):
                    available.append(
                        {
                            "name": m.get("name"),
                            "size": m.get("size"),
                            "family": (m.get("details") or {}).get("family"),
                            "param_size": (m.get("details") or {}).get("parameter_size"),
                        }
                    )
    except Exception:
        # Ollama yetib bormasa ham, config qaytsin
        pass

    return {
        "config": runtime_config.all(),
        "editable_fields": list(EDITABLE_FIELDS.keys()),
        "available_models": available,
        "field_hints": {
            "llm_model": "Chat / generation modeli (masalan: gpt-oss:20b, qwen2.5:14b)",
            "embedding_model": "Embedding modeli (BGE-M3 tavsiya etiladi)",
            "reasoning": "low / medium / high — gpt-oss reasoning effort",
            "llm_temperature": "0.0 (aniq) — 2.0 (ijodiy). Audit uchun 0.1 tavsiya.",
            "llm_max_tokens": "Maksimal javob uzunligi (token)",
            "top_k": "Kontekstga nechta hujjat parchasi olinadi (4 standart)",
            "chunk_size": "Hujjat chunk hajmi (so'zlarda). O'zgartirish — qayta ingest talab qiladi.",
            "chunk_overlap": "Chunklar orasidagi qoplama (so'zlarda)",
            "use_cache": "Semantik keshni yoqish (takroriy savollar uchun)",
            "cache_similarity_threshold": "Kesh ishlashi uchun oʻxshashlik chegarasi (0.95 standart)",
        },
    }


@router.post("")
async def update_config(req: UpdateRequest):
    try:
        old_embedding = runtime_config.get("embedding_model")
        changed = await runtime_config.update(req.updates)
        warnings: list[str] = []

        # Embedding modeli o'zgargan bo'lsa — dimensionni qayta aniqlash
        if "embedding_model" in changed and changed["embedding_model"] != old_embedding:
            try:
                new_dim = await embedder.detect_dim()
                old_dim = vector_store._current_dim
                vector_store.set_current_dim(new_dim)
                if old_dim and old_dim != new_dim:
                    warnings.append(
                        f"⚠️ Yangi embedding modeli boshqa vektor o'lchamiga ega "
                        f"({old_dim} → {new_dim}). Eski kolleksiyalar ishlamaydi. "
                        f"Hujjatlarni qayta indekslash kerak. "
                        f"POST /api/admin/reset_vectors orqali tozalang."
                    )
            except Exception as e:
                warnings.append(f"Embedding dimension aniqlanmadi: {e}")

        return {
            "ok": True,
            "changed": changed,
            "config": runtime_config.all(),
            "warnings": warnings,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset")
async def reset_config():
    await runtime_config.reset()
    return {"ok": True, "config": runtime_config.all()}
