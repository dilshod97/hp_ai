"""Faylni vektor bazaga joylash (chunk -> embed -> upsert)."""
import mimetypes
import os
import uuid
from typing import Any

from app.config import settings
from app.core import documents as docs_svc
from app.core.embeddings import embedder
from app.core.parser import chunk_text, extract_text
from app.core.vector_store import vector_store


async def ingest_file(
    path: str,
    collection: str,
    extra_payload: dict[str, Any] | None = None,
    user_id: int | None = None,
    workspace_id: str | None = None,
) -> dict:
    """Bitta faylni qabul qiladi, chunklarga ajratadi, embed qiladi va Qdrant ga yozadi."""
    text = extract_text(path)
    if not text.strip():
        return {"doc_id": None, "chunks": 0, "skipped": True}

    chunks = list(chunk_text(text))
    if not chunks:
        return {"doc_id": None, "chunks": 0, "skipped": True}

    vectors = await embedder.embed_batch(chunks)
    doc_id = str(uuid.uuid4())
    filename = os.path.basename(path)

    # ❗ Nol vektorli chunklarni o'tkazib yuboramiz — bu embed xatosi
    # (Ollama NaN qaytargan paytda zero-vector qaytarilgan)
    valid_vectors = []
    valid_payloads = []
    skipped_zero = 0
    for i, (vec, ch) in enumerate(zip(vectors, chunks)):
        # Nol vektor — chunkni saqlamasdan o'tkazib yuborish
        if not any(x != 0.0 for x in vec):
            skipped_zero += 1
            continue
        payload = {
            "doc_id": doc_id,
            "source": filename,
            "chunk_index": i,
            "text": ch,
        }
        if user_id is not None:
            payload["user_id"] = user_id
        if extra_payload:
            payload.update(extra_payload)
        valid_vectors.append(vec)
        valid_payloads.append(payload)

    if skipped_zero > 0:
        import logging
        logging.getLogger("ingest").warning(
            "📍 %s: %d/%d chunk nol vektorli edi (embed xatosi), o'tkazib yuborildi",
            filename, skipped_zero, len(chunks),
        )

    if not valid_vectors:
        return {"doc_id": None, "chunks": 0, "skipped": True,
                "error": "Hamma chunklar embedding xato bilan tugadi"}

    await vector_store.upsert(collection, valid_vectors, valid_payloads)

    # Markaziy registrga yozish (Documents jadvali)
    try:
        size_bytes = os.path.getsize(path)
    except OSError:
        size_bytes = 0
    mime, _ = mimetypes.guess_type(filename)
    try:
        await docs_svc.register_document(
            doc_id=doc_id,
            filename=filename,
            user_id=user_id,
            workspace_id=workspace_id,
            collection=collection,
            size_bytes=size_bytes,
            chunks=len(valid_vectors),  # Faqat haqiqiy saqlangan chunks
            mime=mime,
            ocr_used=False,
            file_path=path,
        )
    except Exception:
        # Registry xatosi ingest'ni to'xtatmasin
        pass

    return {
        "doc_id": doc_id, "filename": filename,
        "chunks": len(valid_vectors), "collection": collection,
        "size_bytes": size_bytes,
        "skipped_zero": skipped_zero if skipped_zero else None,
    }


async def ingest_directory(directory: str, collection: str) -> list[dict]:
    """Papkadagi barcha qoʻllab-quvvatlanadigan fayllarni ingest qiladi."""
    results: list[dict] = []
    if not os.path.isdir(directory):
        return results
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".pdf", ".docx", ".xlsx", ".txt", ".md"):
            continue
        try:
            res = await ingest_file(path, collection)
            results.append(res)
        except Exception as e:
            results.append({"filename": name, "error": str(e)})
    return results
