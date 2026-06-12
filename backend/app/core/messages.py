"""Chat xabarlarini DB'ga saqlash va olish."""
from typing import Any

from sqlalchemy import delete, select

from app.core.db import Message, SessionLocal


async def save_message(
    workspace_id: str,
    role: str,
    text: str,
    mode: str | None = None,
    sources: list[Any] | None = None,
    elapsed_ms: int | None = None,
    tokens_per_sec: float | None = None,
    eval_count: int | None = None,
) -> dict:
    async with SessionLocal() as s:
        msg = Message(
            workspace_id=workspace_id,
            role=role,
            text=text,
            mode=mode,
            sources=sources,
            elapsed_ms=elapsed_ms,
            tokens_per_sec=tokens_per_sec,
            eval_count=eval_count,
        )
        s.add(msg)
        await s.commit()
        await s.refresh(msg)
        return msg.to_dict()


async def update_message(message_id: int, **fields) -> dict | None:
    async with SessionLocal() as s:
        msg = await s.get(Message, message_id)
        if not msg:
            return None
        for k, v in fields.items():
            if hasattr(msg, k):
                setattr(msg, k, v)
        await s.commit()
        await s.refresh(msg)
        return msg.to_dict()


async def list_messages(workspace_id: str, limit: int = 200) -> list[dict]:
    async with SessionLocal() as s:
        q = (
            select(Message)
            .where(Message.workspace_id == workspace_id)
            .order_by(Message.id.asc())
            .limit(limit)
        )
        rows = (await s.execute(q)).scalars().all()
        return [r.to_dict() for r in rows]


async def recent_history(workspace_id: str, max_turns: int = 6) -> list[dict]:
    """Oxirgi N juftlik (user+bot) — RAG kontekstida ishlatish uchun."""
    async with SessionLocal() as s:
        q = (
            select(Message)
            .where(Message.workspace_id == workspace_id)
            .where(Message.role.in_(("user", "bot")))
            .order_by(Message.id.desc())
            .limit(max_turns * 2)
        )
        rows = list((await s.execute(q)).scalars().all())
        rows.reverse()
        return [{"role": r.role, "text": r.text} for r in rows if r.text]


async def clear_messages(workspace_id: str) -> int:
    async with SessionLocal() as s:
        q = delete(Message).where(Message.workspace_id == workspace_id)
        result = await s.execute(q)
        await s.commit()
        return result.rowcount or 0


async def rate_message(
    message_id: int, rating: int | None, rated_by: int | None = None
) -> dict | None:
    """Xabarni baholash: 1=like, -1=dislike, None=bahoni olib tashlash."""
    async with SessionLocal() as s:
        msg = await s.get(Message, message_id)
        if not msg or msg.role != "bot":
            return None
        msg.rating = rating
        msg.rated_by = rated_by if rating is not None else None
        await s.commit()
        await s.refresh(msg)
        return msg.to_dict()


async def delete_message(message_id: int) -> bool:
    async with SessionLocal() as s:
        msg = await s.get(Message, message_id)
        if not msg:
            return False
        await s.delete(msg)
        await s.commit()
        return True
