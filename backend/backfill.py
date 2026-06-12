"""Qdrant'dagi mavjud fayllarni Documents jadvalga ko'chirish (bir martalik).

Foydalanish (backend container ichida):
    docker exec hp_backend python scripts/backfill_documents.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa
from app.core.db import SessionLocal, init_db  # noqa
from app.core.documents import register_document  # noqa
from app.core.vector_store import vector_store  # noqa


async def backfill_collection(collection: str) -> int:
    """Bitta collection'dan barcha unique doc_id'larni olib Documents'ga yozish."""
    added = 0
    seen: dict[str, dict] = {}
    offset = None

    while True:
        points, offset = await vector_store.client.scroll(
            collection_name=collection,
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            if doc_id in seen:
                seen[doc_id]["chunks"] += 1
                continue
            seen[doc_id] = {
                "filename": payload.get("source") or f"{doc_id}.unknown",
                "user_id": payload.get("user_id"),
                "chunks": 1,
                "collection": collection,
            }

        if offset is None:
            break

    # DB'ga yozish
    for doc_id, info in seen.items():
        try:
            await register_document(
                doc_id=doc_id,
                filename=info["filename"],
                user_id=info["user_id"],
                workspace_id=None,
                collection=info["collection"],
                size_bytes=0,
                chunks=info["chunks"],
                mime=None,
                ocr_used=False,
            )
            added += 1
        except Exception as e:
            print(f"  ❌ {doc_id}: {e}")

    return added


async def main():
    print("=" * 50)
    print("📦 Documents backfill (Qdrant → SQLite)")
    print("=" * 50)
    await init_db()
    total = 0
    for col in (
        settings.QDRANT_COLLECTION_UPLOADS,
        settings.QDRANT_COLLECTION_LAWS,
        settings.QDRANT_COLLECTION_REPORTS,
    ):
        print(f"\n→ Collection: {col}")
        try:
            n = await backfill_collection(col)
            print(f"  ✓ {n} ta hujjat ro'yxatga oldi")
            total += n
        except Exception as e:
            print(f"  ❌ Xato: {e}")

    print()
    print("=" * 50)
    print(f"✅ Tayyor! Jami {total} ta hujjat backfill qilindi")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
