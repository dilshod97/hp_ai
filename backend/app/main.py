"""Hisob palatasi AI — FastAPI kirish nuqtasi."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    admin,
    all_documents,
    auth as auth_api,
    chat,
    compare,
    config as config_api,
    cross,
    documents,
    finetune,
    health,
    models,
    monitoring,
    users as users_api,
    workspaces as workspaces_api,
)
from app.config import settings
from app.core.auth import ensure_admin
from app.core.db import init_db
from app.core.embeddings import embedder
from app.core.llm import llm
from app.core.ollama_init import ensure_models
from app.core.vector_store import vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("hp_ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 HP-AI audit assistant ishga tushyapti...")

    # 0) SQLite tayyorlash (chat tarixi uchun)
    try:
        await init_db()
        await ensure_admin()
    except Exception as e:
        log.error("DB init xatosi: %s", e)

    # 1) Ollama modellarini avtomatik tortib olish (LLM + Embedding)
    try:
        await ensure_models()
    except Exception as e:
        log.error("ensure_models xatosi: %s", e)

    # 2) Embedding modeli dimensionini aniqlash (Qdrant collection o'lchami uchun)
    try:
        dim = await embedder.detect_dim()
        vector_store.set_current_dim(dim)
        log.info("✓ Embedding dimension aniqlandi: %s = %d", embedder.model, dim)
    except Exception as e:
        log.warning("Embedding dimension aniqlanmadi: %s — fallback 1024", e)
        vector_store.set_current_dim(1024)

    # 3) Qdrant collectionlarini tayyorlash (joriy oʻlcham bilan)
    for col in (
        settings.QDRANT_COLLECTION_LAWS,
        settings.QDRANT_COLLECTION_REPORTS,
        settings.QDRANT_COLLECTION_UPLOADS,
    ):
        try:
            await vector_store.ensure_collection(col)
            log.info("✓ Qdrant collection tayyor: %s", col)
        except Exception as e:
            log.warning("Qdrant collection xatosi (%s): %s", col, e)

    log.info("✅ Backend tayyor — http://localhost:8000/docs")
    yield

    log.info("👋 Backend toʻxtatilyapti...")
    await embedder.close()
    await llm.close()
    await vector_store.close()


app = FastAPI(
    title="HP-AI audit assistant",
    description="Hisob palatasi auditorlari uchun RAG asosida savol-javob tizimi",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(compare.router)
app.include_router(models.router)
app.include_router(config_api.router)
app.include_router(admin.router)
app.include_router(workspaces_api.router)
app.include_router(auth_api.router)
app.include_router(users_api.router)
app.include_router(cross.router)
app.include_router(monitoring.router)
app.include_router(finetune.router)
app.include_router(all_documents.router)


@app.get("/")
async def root():
    return {
        "name": "HP-AI audit assistant",
        "version": "1.0.0",
        "description": "Hisob palatasi auditorlari uchun yordamchi sun'iy intellekt",
        "docs": "/docs",
        "health": "/api/health",
    }
