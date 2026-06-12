"""Markaziy hujjatlar registri:
- Foydalanuvchi o'zining yuklagan fayllarini ko'radi
- Admin — barchaning fayllarini koʻradi
"""
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import settings
from app.core import documents as docs_svc
from app.core.auth import current_user, current_user_optional, decode_token, require_admin
from app.core.db import SessionLocal, User
from app.core.vector_store import vector_store

router = APIRouter(prefix="/api/all_documents", tags=["documents-registry"])


async def _fetch_full_text(doc_id: str, collection: str) -> tuple[str, str | None]:
    """Berilgan doc_id ning barcha chunklarini Qdrant'dan olib, matnga jamlash."""
    chunks: list[tuple[int, str]] = []
    filename: str | None = None
    offset = None
    while True:
        points, offset = await vector_store.client.scroll(
            collection_name=collection,
            scroll_filter={
                "must": [{"key": "doc_id", "match": {"value": doc_id}}]
            },
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            idx = payload.get("chunk_index", 0)
            text = payload.get("text", "")
            if text:
                chunks.append((idx, text))
            if not filename and payload.get("source"):
                filename = payload["source"]
        if offset is None:
            break
    chunks.sort(key=lambda x: x[0])
    return "\n\n".join(c[1] for c in chunks), filename


@router.get("")
async def list_documents(
    search: str | None = None,
    user_id: int | None = None,
    limit: int = 500,
    user: dict = Depends(current_user),
):
    """Fayllar ro'yxati.

    - User: faqat o'zinikini
    - Admin: barchasini (user_id filter bilan tanlash mumkin)
    """
    if user.get("role") == "admin":
        items = await docs_svc.list_documents(
            user_id=user_id, limit=limit, search=search
        )
    else:
        items = await docs_svc.list_documents(
            user_id=user["id"], limit=limit, search=search
        )
    return {"items": items, "count": len(items)}


@router.get("/stats")
async def documents_stats(_: dict = Depends(require_admin)):
    """Umumiy statistika (admin only): hajm, ocr soni, foydalanuvchilar boʻyicha."""
    return await docs_svc.stats()


async def _user_from_token_param(token: str | None) -> dict | None:
    """Query parameter'dagi tokendan user'ni olish (download URL'lar uchun)."""
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError):
        return None
    async with SessionLocal() as s:
        u = await s.get(User, user_id)
        if not u or not u.is_active:
            return None
        return u.to_dict()


@router.get("/{doc_id}/download")
async def download_document(
    doc_id: str,
    token: str | None = Query(None, description="JWT token (img tag uchun)"),
    user: dict | None = Depends(current_user_optional),
):
    """Asl faylni yuklab olish yoki preview (rasm uchun img tag'da koʻrsatish).

    Token query param'da ham qabul qilinadi — rasm tag'larida Authorization header
    yuborib bo'lmasligi sababli.
    """
    actual_user = user or await _user_from_token_param(token)
    if not actual_user:
        raise HTTPException(status_code=401, detail="Tizimga kiring")

    doc = await docs_svc.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Hujjat topilmadi")
    if actual_user.get("role") != "admin" and doc.get("user_id") != actual_user.get("id"):
        raise HTTPException(status_code=403, detail="Ruxsat yo'q")

    path = doc.get("file_path")
    if not path or not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Fayl topilmadi (eski yuklama bo'lishi mumkin)",
        )

    return FileResponse(
        path,
        filename=doc.get("filename"),
        media_type=doc.get("mime") or "application/octet-stream",
    )


@router.get("/{doc_id}/text")
async def get_document_text(doc_id: str, user: dict = Depends(current_user)):
    """Hujjatning to'liq matnini qaytaradi (chunks birlashtirilgan)."""
    doc = await docs_svc.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Hujjat topilmadi")
    if user.get("role") != "admin" and doc.get("user_id") != user.get("id"):
        raise HTTPException(status_code=403, detail="Ruxsat yo'q")

    text, filename = await _fetch_full_text(
        doc_id, doc.get("collection") or settings.QDRANT_COLLECTION_UPLOADS
    )
    return {
        "doc_id": doc_id,
        "filename": filename or doc.get("filename"),
        "text": text,
        "chars": len(text),
        "chunks": doc.get("chunks"),
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, user: dict = Depends(current_user)):
    """Hujjatni o'chirish — Qdrant'dan ham, DB'dan ham."""
    doc = await docs_svc.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Hujjat topilmadi")
    # Faqat egasi yoki admin o'chira oladi
    if user.get("role") != "admin" and doc.get("user_id") != user.get("id"):
        raise HTTPException(status_code=403, detail="Ruxsat yo'q")

    # 1) Qdrant'dan
    try:
        await vector_store.delete_by_doc(
            doc.get("collection") or settings.QDRANT_COLLECTION_UPLOADS,
            doc_id,
        )
    except Exception:
        pass
    # 2) DB'dan
    await docs_svc.delete_document(doc_id)
    return {"ok": True}
