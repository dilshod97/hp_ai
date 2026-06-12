"""Training job CRUD va background runner.

Training serverda alohida jarayonda ishlaydi (subprocess). Backend faqat:
- Job yaratadi (DB'ga)
- Jarayonni boshlaydi
- Stdout'ni log_tail'ga yozadi
- Progressni yangilaydi
"""
import asyncio
import json
import logging
import os
import shlex
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.core.datasets import export_jsonl
from app.core.db import SessionLocal, TrainingJob

log = logging.getLogger("training")

MODELS_DIR = os.path.join(settings.DATA_DIR, "trained_models")
TRAINING_SCRIPT = os.path.join(settings.DATA_DIR, "scripts", "train.py")

# Running jarayonlar — job_id -> Process
_running: dict[int, asyncio.subprocess.Process] = {}


# ---------- CRUD ----------

async def create_job(
    dataset_id: int,
    base_model: str,
    output_model_name: str,
    config: dict[str, Any] | None = None,
) -> dict:
    async with SessionLocal() as s:
        job = TrainingJob(
            dataset_id=dataset_id,
            base_model=base_model,
            output_model_name=output_model_name,
            config=config or {},
            status="pending",
            progress=0.0,
        )
        s.add(job)
        await s.commit()
        await s.refresh(job)
        return job.to_dict()


async def list_jobs() -> list[dict]:
    async with SessionLocal() as s:
        rows = (await s.execute(
            select(TrainingJob).order_by(TrainingJob.id.desc()).limit(50)
        )).scalars().all()
        return [r.to_dict() for r in rows]


async def get_job(job_id: int) -> dict | None:
    async with SessionLocal() as s:
        j = await s.get(TrainingJob, job_id)
        return j.to_dict() if j else None


async def update_job(job_id: int, **fields) -> None:
    async with SessionLocal() as s:
        j = await s.get(TrainingJob, job_id)
        if not j:
            return
        for k, v in fields.items():
            if hasattr(j, k):
                setattr(j, k, v)
        await s.commit()


async def cancel_job(job_id: int) -> bool:
    proc = _running.get(job_id)
    if proc and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
        await update_job(job_id, status="failed", error="Bekor qilindi")
        return True
    return False


# ---------- Runner ----------

async def start_job(job_id: int) -> None:
    """Background da training boshlash."""
    job = await get_job(job_id)
    if not job:
        return
    # Dataset'ni JSONL ga eksport qilamiz
    try:
        jsonl_path = await export_jsonl(job["dataset_id"])
    except Exception as e:
        await update_job(job_id, status="failed", error=f"JSONL eksport xatosi: {e}")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    output_dir = os.path.join(MODELS_DIR, job["output_model_name"])

    # Training skript: scripts/train.py
    cfg = job.get("config") or {}
    cmd = [
        "python", "/app/scripts/train.py",
        "--dataset", jsonl_path,
        "--base-model", job["base_model"],
        "--output", output_dir,
        "--epochs", str(cfg.get("epochs", 3)),
        "--lr", str(cfg.get("learning_rate", 2e-4)),
        "--batch-size", str(cfg.get("batch_size", 2)),
        "--lora-r", str(cfg.get("lora_r", 16)),
    ]

    await update_job(
        job_id, status="running", progress=0.05,
        log_tail=f"$ {' '.join(shlex.quote(c) for c in cmd)}\n",
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        _running[job_id] = proc

        log_buffer: list[str] = []
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="ignore").rstrip()
            log_buffer.append(text)
            # Progress parsing — train.py "PROGRESS 0.45" formatida chiqarsa
            if text.startswith("PROGRESS"):
                try:
                    pct = float(text.split()[1])
                    await update_job(job_id, progress=pct)
                except Exception:
                    pass
            # Loglarni har 10 satrda DB'ga yozish (overhead kamaytirish)
            if len(log_buffer) % 10 == 0:
                tail = "\n".join(log_buffer[-200:])
                await update_job(job_id, log_tail=tail)

        await proc.wait()
        _running.pop(job_id, None)
        tail = "\n".join(log_buffer[-200:])

        if proc.returncode == 0:
            await update_job(
                job_id, status="completed", progress=1.0,
                log_tail=tail, finished_at=datetime.utcnow(),
            )
            log.info("✅ Training tugadi: job %d", job_id)
        else:
            await update_job(
                job_id, status="failed",
                error=f"Process exit code: {proc.returncode}",
                log_tail=tail, finished_at=datetime.utcnow(),
            )
    except FileNotFoundError as e:
        await update_job(
            job_id, status="failed",
            error=f"Training skript topilmadi: {e}. "
                  f"scripts/train.py ni ko'rib chiqing.",
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        log.exception("Training xato: %s", e)
        await update_job(
            job_id, status="failed", error=str(e),
            finished_at=datetime.utcnow(),
        )
