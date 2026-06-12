"""Admin: vector kolleksiyalarni reset qilish, keshni tozalash va boshqa servis amallari."""
from fastapi import APIRouter, HTTPException

from app.core.cache import cache
from app.core.embeddings import embedder
from app.core.vector_store import vector_store

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/reset_vectors")
async def reset_vectors():
    """Barcha Qdrant kolleksiyalarni o'chiradi va joriy dimension bilan qayta yaratadi.

    Embedding model o'zgartirilgandan keyin ishlatiladi.
    Bu amaldan keyin hujjatlarni qayta ingest qilish kerak.
    """
    try:
        # Joriy dimensionni qayta aniqlash
        dim = await embedder.detect_dim()
        vector_store.set_current_dim(dim)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding olishda xato: {e}")

    deleted = await vector_store.delete_all()
    # ensure_collection mexanizmi keyingi murojaatda yangi dim bilan qayta yaratadi
    return {
        "ok": True,
        "embedding_model": embedder.model,
        "new_dim": dim,
        "deleted": deleted,
        "next_step": "Hujjatlarni qayta yuklang yoki ingest qiling.",
    }


@router.post("/clear_cache")
async def clear_cache():
    """Semantik keshni butunlay tozalash."""
    try:
        await cache.client.flushdb()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
