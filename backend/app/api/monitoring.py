"""Monitoring: Ollama loaded models, GPU usage, backend metrics.

GPU statistikasi uchun 2 ta variant:
1. Server'da nvidia_gpu_exporter ishlasa — uning /metrics URL (.env: GPU_STATS_URL)
2. Server'da /api/gpu kabi maxsus agent boʻlsa — JSON qaytaradi
"""
import asyncio
import json
import logging
import re
from typing import Any

import httpx
import psutil
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.auth import current_user
from app.core.db import SessionLocal, Message, User
from sqlalchemy import select, func

log = logging.getLogger("monitoring")
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


# ---------- Ollama ----------

async def _ollama_ps() -> list[dict]:
    """Hozir RAM/VRAM ga yuklangan modellar."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{settings.OLLAMA_BASE_URL}/api/ps")
            r.raise_for_status()
            return r.json().get("models", [])
    except Exception as e:
        log.warning("Ollama /ps xato: %s", e)
        return []


async def _ollama_tags() -> list[dict]:
    """Diskdagi barcha modellar."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
            return r.json().get("models", [])
    except Exception:
        return []


# ---------- GPU (ixtiyoriy) ----------

_PROM_PATTERNS = {
    "gpu_util": re.compile(r"DCGM_FI_DEV_GPU_UTIL\{[^}]*\}\s+([\d.]+)"),
    "mem_used": re.compile(r"DCGM_FI_DEV_FB_USED\{[^}]*\}\s+([\d.]+)"),
    "mem_total": re.compile(r"DCGM_FI_DEV_FB_FREE\{[^}]*\}\s+([\d.]+)"),
    "temp": re.compile(r"DCGM_FI_DEV_GPU_TEMP\{[^}]*\}\s+([\d.]+)"),
    "power": re.compile(r"DCGM_FI_DEV_POWER_USAGE\{[^}]*\}\s+([\d.]+)"),
    # nvidia_gpu_exporter formatlari ham (alternativ)
    "alt_util": re.compile(r"nvidia_gpu_utilization_gpu_ratio\s+([\d.]+)"),
    "alt_mem_used": re.compile(r"nvidia_gpu_memory_used_bytes\s+([\d.]+)"),
    "alt_mem_total": re.compile(r"nvidia_gpu_memory_total_bytes\s+([\d.]+)"),
    "alt_temp": re.compile(r"nvidia_gpu_temperature_celsius\s+([\d.]+)"),
}


async def _gpu_stats() -> dict[str, Any] | None:
    url = settings.GPU_STATS_URL
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(url)
            r.raise_for_status()
            text = r.text
        # JSON formatini ham sinaymiz
        if text.strip().startswith("{"):
            try:
                return json.loads(text)
            except Exception:
                pass
        # Prometheus matni
        result: dict[str, Any] = {}
        for k, pat in _PROM_PATTERNS.items():
            m = pat.search(text)
            if m:
                result[k] = float(m.group(1))
        return result if result else None
    except Exception as e:
        log.warning("GPU stats xato: %s", e)
        return None


# ---------- Backend metrics ----------

async def _backend_metrics() -> dict[str, Any]:
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_percent": cpu,
            "memory_used_gb": round(mem.used / 1024**3, 2),
            "memory_total_gb": round(mem.total / 1024**3, 2),
            "memory_percent": mem.percent,
            "disk_used_gb": round(disk.used / 1024**3, 2),
            "disk_total_gb": round(disk.total / 1024**3, 2),
            "disk_percent": disk.percent,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------- DB statistika ----------

async def _db_stats() -> dict[str, Any]:
    async with SessionLocal() as s:
        users_count = (await s.execute(select(func.count(User.id)))).scalar_one()
        msgs_count = (await s.execute(select(func.count(Message.id)))).scalar_one()
        # Oxirgi 24 soat
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent = (await s.execute(
            select(func.count(Message.id)).where(Message.created_at >= cutoff)
        )).scalar_one()
        return {
            "users": users_count,
            "messages_total": msgs_count,
            "messages_last_24h": recent,
        }


# ---------- Endpoints ----------

@router.get("/snapshot")
async def snapshot(_: dict = Depends(current_user)):
    """Birinchi marta — barcha holatni qaytaradi."""
    ps, tags, gpu, sys, db = await asyncio.gather(
        _ollama_ps(),
        _ollama_tags(),
        _gpu_stats(),
        _backend_metrics(),
        _db_stats(),
    )
    return {
        "ollama": {
            "base_url": settings.OLLAMA_BASE_URL,
            "loaded_models": ps,
            "available_models": tags,
        },
        "gpu": gpu,
        "system": sys,
        "db": db,
    }


@router.get("/stream")
async def stream(_: dict = Depends(current_user)):
    """SSE — har 3 sekundda yangi snapshot."""

    async def gen():
        while True:
            ps, gpu, sys = await asyncio.gather(
                _ollama_ps(), _gpu_stats(), _backend_metrics()
            )
            payload = {"loaded_models": ps, "gpu": gpu, "system": sys}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(3)

    return StreamingResponse(gen(), media_type="text/event-stream")
