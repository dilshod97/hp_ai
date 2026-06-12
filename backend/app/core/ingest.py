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

    payloads = []
    for i, ch in enumerate(chunks):
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
        payloads.append(payload)

    await vector_store.upsert(collection, vectors, payloads)

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
            chunks=len(chunks),
            mime=mime,
            ocr_used=False,
            file_path=path,  # Asl fayl saqlangan joy (download/preview uchun)
        )
    except Exception:
        # Registry xatosi ingest'ni to'xtatmasin
        pass

    return {
        "doc_id": doc_id, "filename": filename,
        "chunks": len(chunks), "collection": collection,
        "size_bytes": size_bytes,
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
