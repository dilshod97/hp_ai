"""Hujjat yuklash va ingest qilish endpointlari."""
import os
from typing import Literal

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import settings
from app.core.auth import current_user, require_admin
from app.core.ingest import ingest_directory, ingest_file
from app.models.schemas import IngestResponse

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXT = {
    ".pdf", ".docx", ".xlsx", ".txt", ".md",
    # OCR uchun rasm formatlari (Tesseract bilan)
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
}


def _resolve_collection(target: str) -> str:
    if target == "laws":
        return settings.QDRANT_COLLECTION_LAWS
    if target == "reports":
        return settings.QDRANT_COLLECTION_REPORTS
    return settings.QDRANT_COLLECTION_UPLOADS


@router.post("/upload", response_model=IngestResponse)
async def upload(
    file: UploadFile = File(...),
    target: Literal["uploads", "laws", "reports"] = Form("uploads"),
    user: dict = Depends(current_user),
):
    """Fayl yuklash.

    target='uploads' — shaxsiy fayl (foydalanuvchi)
    target='laws' / 'reports' — global bilim bazasi (faqat admin)
    """
    # laws/reports — global bilim bazasi: faqat admin
    if target in ("laws", "reports") and user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Faqat admin asosiy bilim bazasiga fayl qoʻsha oladi",
        )

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Qoʻllab-quvvatlanmaydigan format: {ext}",
        )

    # Bilim bazasi fayllari alohida papkalarga
    if target == "laws":
        dest_dir = settings.LAWS_DIR
    elif target == "reports":
        dest_dir = settings.REPORTS_DIR
    else:
        dest_dir = settings.UPLOAD_DIR
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, file.filename)
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    try:
        # Global fayl (laws/reports) — user_id=None (hammasi koʻradi)
        uid = None if target in ("laws", "reports") else user.get("id")
        result = await ingest_file(
            dest, _resolve_collection(target),
            user_id=uid,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest xatosi: {e}")


@router.post("/ingest_dir")
async def ingest_dir(
    target: Literal["uploads", "laws", "reports"] = Form(...),
    path: str | None = Form(None),
):
    """Server ichidagi papkani ingest qilish (laws/ yoki reports/)."""
    if not path:
        path = (
            settings.LAWS_DIR
            if target == "laws"
            else settings.REPORTS_DIR
            if target == "reports"
            else settings.UPLOAD_DIR
        )
    results = await ingest_directory(path, _resolve_collection(target))
    return {"items": results, "count": len(results)}
