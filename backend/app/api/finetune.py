"""Fine-tuning API: datasetlar va training jobs.

Faqat admin uchun (resource-intensive).
"""
import asyncio
import os
from typing import Any, Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.core import datasets as ds_svc
from app.core import training as tr_svc
from app.core.auth import require_admin

router = APIRouter(prefix="/api/finetune", tags=["finetune"])


# ---------- Sxemalar ----------

class CreateDatasetBody(BaseModel):
    name: str
    description: Optional[str] = None


class ItemBody(BaseModel):
    instruction: str
    output: str
    input: Optional[str] = None
    quality: int = 3


class UpdateItemBody(BaseModel):
    instruction: Optional[str] = None
    output: Optional[str] = None
    input: Optional[str] = None
    quality: Optional[int] = None


class ImportFromChatBody(BaseModel):
    workspace_id: Optional[str] = None
    only_good: bool = True


class StartJobBody(BaseModel):
    dataset_id: int
    base_model: str = "gpt-oss:20b"
    output_model_name: str
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    lora_r: int = 16


# ---------- Datasets ----------

@router.get("/datasets")
async def list_datasets(_: dict = Depends(require_admin)):
    return {"items": await ds_svc.list_datasets()}


@router.post("/datasets")
async def create_dataset(body: CreateDatasetBody, user: dict = Depends(require_admin)):
    return await ds_svc.create_dataset(body.name, body.description, owner_id=user.get("id"))


@router.delete("/datasets/{ds_id}")
async def delete_dataset(ds_id: int, _: dict = Depends(require_admin)):
    ok = await ds_svc.delete_dataset(ds_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Dataset topilmadi")
    return {"ok": True}


@router.get("/datasets/{ds_id}/items")
async def list_items(ds_id: int, limit: int = 500, _: dict = Depends(require_admin)):
    return {"items": await ds_svc.list_items(ds_id, limit=limit)}


@router.post("/datasets/{ds_id}/items")
async def add_item(ds_id: int, body: ItemBody, _: dict = Depends(require_admin)):
    return await ds_svc.add_item(
        ds_id, body.instruction, body.output, body.input, quality=body.quality
    )


@router.patch("/datasets/{ds_id}/items/{item_id}")
async def update_item(
    ds_id: int, item_id: int, body: UpdateItemBody, _: dict = Depends(require_admin)
):
    fields = body.model_dump(exclude_none=True)
    item = await ds_svc.update_item(item_id, **fields)
    if not item:
        raise HTTPException(status_code=404, detail="Item topilmadi")
    return item


@router.delete("/datasets/{ds_id}/items/{item_id}")
async def delete_item(ds_id: int, item_id: int, _: dict = Depends(require_admin)):
    await ds_svc.delete_item(item_id)
    return {"ok": True}


@router.post("/datasets/{ds_id}/import_chat")
async def import_from_chat(
    ds_id: int, body: ImportFromChatBody, _: dict = Depends(require_admin)
):
    n = await ds_svc.import_from_chat(ds_id, body.workspace_id, body.only_good)
    return {"added": n}


@router.post("/datasets/{ds_id}/import_jsonl")
async def import_jsonl(
    ds_id: int, file: UploadFile = File(...), _: dict = Depends(require_admin)
):
    upload_dir = os.path.join(settings.DATA_DIR, "datasets", "_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, file.filename or "uploaded.jsonl")
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)
    n = await ds_svc.import_jsonl(ds_id, dest)
    return {"added": n, "file": file.filename}


@router.get("/datasets/{ds_id}/export")
async def export_jsonl(ds_id: int, _: dict = Depends(require_admin)):
    path = await ds_svc.export_jsonl(ds_id)
    return FileResponse(path, filename=os.path.basename(path), media_type="application/x-jsonlines")


# ---------- Training jobs ----------

@router.get("/jobs")
async def list_jobs(_: dict = Depends(require_admin)):
    return {"items": await tr_svc.list_jobs()}


@router.get("/jobs/{job_id}")
async def get_job(job_id: int, _: dict = Depends(require_admin)):
    job = await tr_svc.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Topilmadi")
    return job


@router.post("/jobs")
async def start_training(
    body: StartJobBody, bg: BackgroundTasks, _: dict = Depends(require_admin)
):
    job = await tr_svc.create_job(
        body.dataset_id,
        body.base_model,
        body.output_model_name,
        config={
            "epochs": body.epochs,
            "learning_rate": body.learning_rate,
            "batch_size": body.batch_size,
            "lora_r": body.lora_r,
        },
    )
    # Background da boshlaymiz
    asyncio.create_task(tr_svc.start_job(job["id"]))
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, _: dict = Depends(require_admin)):
    ok = await tr_svc.cancel_job(job_id)
    return {"cancelled": ok}
