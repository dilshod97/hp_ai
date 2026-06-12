"""SQLite + SQLAlchemy async setup.

DB fayl: /data/hp_ai.db  (volume orqali host'ga saqlanadi)
"""
import logging
import os
from datetime import datetime
from typing import AsyncIterator

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings

log = logging.getLogger("db")

DB_PATH = os.path.join(settings.DATA_DIR, "hp_ai.db")
DB_URL = f"sqlite+aiosqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)  # soha (masalan: "Moliya", "Davlat xaridlari")
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), default="user", nullable=False)  # admin / user
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self, include_hash: bool = False) -> dict:
        d = {
            "id": self.id,
            "username": self.username,
            "full_name": self.full_name,
            "sector": self.sector,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_hash:
            d["password_hash"] = self.password_hash
        return d


class Document(Base):
    """Yuklangan hujjatlar markaziy ro'yxati.

    Qdrant'dagi chunks bilan bog'lanadi (doc_id), lekin metadata bu yerda.
    """
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    workspace_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    collection: Mapped[str] = mapped_column(String(64), nullable=False, default="uploads")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mime: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ocr_used: Mapped[bool] = mapped_column(default=False, nullable=False)
    # Faylning saqlangan joyi (download/preview uchun)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True, nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "collection": self.collection,
            "size_bytes": self.size_bytes,
            "chunks": self.chunks,
            "mime": self.mime,
            "ocr_used": self.ocr_used,
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DatasetItem(Base):
    __tablename__ = "dataset_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    # Standart instruct format: instruction + input -> output
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    output: Mapped[str] = mapped_column(Text, nullable=False)
    # Manba — chat history'dan kelsa, message ID, qo'lda kiritilgan bo'lsa null
    source_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality: Mapped[int] = mapped_column(Integer, default=3, nullable=False)  # 1-5
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output,
            "source_message_id": self.source_message_id,
            "quality": self.quality,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False)
    base_model: Mapped[str] = mapped_column(String(128), nullable=False)
    output_model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    # pending / running / completed / failed
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    log_tail: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "base_model": self.base_model,
            "output_model_name": self.output_model_name,
            "status": self.status,
            "config": self.config,
            "progress": self.progress,
            "log_tail": self.log_tail,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user / bot / attach
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # rag / llm_only / no_context
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_per_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True, nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "role": self.role,
            "text": self.text,
            "mode": self.mode,
            "sources": self.sources or [],
            "elapsed_ms": self.elapsed_ms,
            "tokens_per_sec": self.tokens_per_sec,
            "eval_count": self.eval_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Engine va Session factory
engine = create_async_engine(DB_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("✓ SQLite tayyor: %s", DB_PATH)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
