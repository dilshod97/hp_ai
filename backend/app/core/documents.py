"""Document CRUD — yuklangan fayllar markaziy registri."""
from typing import Any

from sqlalchemy import desc, func, select

from app.core.db import Document, SessionLocal, User


async def register_document(
    doc_id: str,
    filename: str,
    user_id: int | None = None,
    workspace_id: str | None = None,
    collection: str = "uploads",
    size_bytes: int = 0,
    chunks: int = 0,
    mime: str | None = None,
    ocr_used: bool = False,
    file_path: str | None = None,
) -> dict:
    """Yangi hujjatni DB'ga ro'yxatdan o'tkazish (upload paytida chaqiriladi)."""
    async with SessionLocal() as s:
        # Mavjudligini tekshirish (idempotent)
        existing = (await s.execute(
            select(Document).where(Document.doc_id == doc_id)
        )).scalar_one_or_none()
        if existing:
            return existing.to_dict()

        doc = Document(
            doc_id=doc_id,
            filename=filename,
            user_id=user_id,
            workspace_id=workspace_id,
            collection=collection,
            size_bytes=size_bytes,
            chunks=chunks,
            mime=mime,
            ocr_used=ocr_used,
            file_path=file_path,
        )
        s.add(doc)
        await s.commit()
        await s.refresh(doc)
        return doc.to_dict()


async def list_documents(
    user_id: int | None = None,
    limit: int = 500,
    search: str | None = None,
) -> list[dict]:
    """Hujjatlar ro'yxati. user_id=None bo'lsa — barchasini qaytaradi (admin).

    Har bir hujjatga foydalanuvchi ma'lumoti ham qoʻshiladi.
    """
    async with SessionLocal() as s:
        q = select(Document, User).outerjoin(User, Document.user_id == User.id)
        if user_id is not None:
            q = q.where(Document.user_id == user_id)
        if search:
            like = f"%{search.lower()}%"
            q = q.where(func.lower(Document.filename).like(like))
        q = q.order_by(desc(Document.created_at)).limit(limit)

        rows = (await s.execute(q)).all()
        result = []
        for doc, user in rows:
            item = doc.to_dict()
            if user:
                item["user"] = {
                    "id": user.id,
                    "username": user.username,
                    "full_name": user.full_name,
                    "sector": user.sector,
                }
            else:
                item["user"] = None
            result.append(item)
        return result


async def get_document(doc_id: str) -> dict | None:
    async with SessionLocal() as s:
        d = (await s.execute(
            select(Document).where(Document.doc_id == doc_id)
        )).scalar_one_or_none()
        return d.to_dict() if d else None


async def delete_document(doc_id: str) -> bool:
    async with SessionLocal() as s:
        d = (await s.execute(
            select(Document).where(Document.doc_id == doc_id)
        )).scalar_one_or_none()
        if not d:
            return False
        await s.delete(d)
        await s.commit()
        return True


async def stats() -> dict:
    """Umumiy statistika."""
    async with SessionLocal() as s:
        total = (await s.execute(select(func.count(Document.id)))).scalar_one()
        total_size = (await s.execute(select(func.coalesce(func.sum(Document.size_bytes), 0)))).scalar_one()
        total_chunks = (await s.execute(select(func.coalesce(func.sum(Document.chunks), 0)))).scalar_one()
        ocr_count = (await s.execute(
            select(func.count(Document.id)).where(Document.ocr_used == True)  # noqa
        )).scalar_one()
        # Foydalanuvchilar bo'yicha
        by_user = (await s.execute(
            select(
                User.username, User.full_name, User.sector,
                func.count(Document.id).label("docs_count"),
                func.coalesce(func.sum(Document.size_bytes), 0).label("total_size"),
            )
            .outerjoin(Document, Document.user_id == User.id)
            .group_by(User.id)
            .order_by(desc("docs_count"))
        )).all()
        return {
            "total": total,
            "total_size_bytes": total_size,
            "total_chunks": total_chunks,
            "ocr_count": ocr_count,
            "by_user": [
                {
                    "username": r[0], "full_name": r[1], "sector": r[2],
                    "docs": r[3], "size_bytes": r[4],
                }
                for r in by_user
            ],
        }
