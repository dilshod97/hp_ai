"""Workspace endpointlari + per-workspace ask/upload (avtomatik API).

Chat tarixi SQLite'ga saqlanadi (frontend yuborishi shart emas).
"""
import json
import os
import re
import time
from typing import Any, Optional

# "bu file", "shu hujjat", "yuklangan fayl" — oxirgi faylga ishora qiluvchi so'zlar
_THIS_DOC_PATTERNS = re.compile(
    r"\b(bu|shu|ushbu|yuklagan|yuklangan|oxirgi|hozirgi|mana shu)"
    r"\s+(file|fayl|hujjat|hujjatda|fileda|faylda|hujjatni|faylni|filein|hujjatim|faylim|filim|hisobotda|hisobot|qog'oz)"
    r"|file\s+nima\s+haqida|fayl\s+nima\s+haqida|hujjat\s+nima\s+haqida"
    r"|bu\s+nima\s+haqida",
    re.IGNORECASE,
)

# "tekstini ber", "matnini ko'rsat", "ichidagi yozuvni" — to'liq matn so'rovi
_FULL_TEXT_PATTERNS = re.compile(
    r"\b(tekstini|matnini|matn|tekst|yozuvni|ichidagini|ichidagi\s+yozuv|to\'liq\s+matn|toʻliq\s+matn|full\s+text)"
    r"\s*(ber|ko'rsat|koʻrsat|chiqar|olib\s+ber|ol)?|tekst\s+ber|matn\s+ber"
    r"|ichidagi\s+yozuvni\s+ber|nima\s+yozilgan",
    re.IGNORECASE,
)


def _wants_recent_doc(question: str) -> bool:
    """Foydalanuvchi 'bu file/shu hujjat' degan tarzda oxirgi faylni so'rayaptimi?"""
    return bool(_THIS_DOC_PATTERNS.search(question or ""))


def _wants_full_text(question: str) -> bool:
    """'Tekstini ber', 'matnini ko'rsat' kabi to'liq matn so'rovi."""
    return bool(_FULL_TEXT_PATTERNS.search(question or ""))

import aiofiles
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.core import messages as msgs
from app.core import rag
from app.core.auth import current_user, current_user_optional
from app.core.export import export_docx, export_pdf
from app.core.ingest import ingest_file
from app.core.vector_store import vector_store
from app.core.workspaces import workspaces


async def _full_text_of(doc_id: str, collection: str) -> str:
    """Bitta doc_id ning chunklarini birlashtirib to'liq matn qaytarish."""
    chunks: list[tuple[int, str]] = []
    offset = None
    while True:
        points, offset = await vector_store.client.scroll(
            collection_name=collection,
            scroll_filter={
                "must": [{"key": "doc_id", "match": {"value": doc_id}}]
            },
            limit=500,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            chunks.append((payload.get("chunk_index", 0), payload.get("text", "")))
        if offset is None:
            break
    chunks.sort(key=lambda x: x[0])
    return "\n\n".join(c[1] for c in chunks if c[1])

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


# ---------- Schemas ----------

class WorkspaceCreate(BaseModel):
    name: str
    icon: Optional[str] = "💬"
    scope: Optional[list[str]] = None
    system_prompt: Optional[str] = ""
    allow_fallback: Optional[bool] = True
    doc_ids: Optional[list[str]] = None


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = None
    icon: Optional[str] = None
    scope: Optional[list[str]] = None
    system_prompt: Optional[str] = None
    allow_fallback: Optional[bool] = None
    doc_ids: Optional[list[str]] = None


class HistoryItem(BaseModel):
    role: str  # "user" yoki "bot"
    text: str


class AskBody(BaseModel):
    question: str
    history: Optional[list[HistoryItem]] = None
    # Shu so'rovda yuklangan fayllarning doc_id'lari — eng yuqori prioritet
    active_doc_ids: Optional[list[str]] = None


# ---------- Yordamchi ----------

def _resolve(
    wid: str,
    api_key: str | None,
    user: dict | None = None,
) -> dict[str, Any]:
    """Workspace'ga kirish — quyidagilardan biri kerak:
    - API key mos kelishi (tashqi integratsiya)
    - Login qilgan + workspace'ning egasi yoki admin
    """
    ws = workspaces.get(wid)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace topilmadi")

    if api_key:
        if api_key == ws["api_key"]:
            return ws
        raise HTTPException(status_code=401, detail="Noto'g'ri API key")

    if not user:
        raise HTTPException(status_code=401, detail="Tizimga kiring")

    if user.get("role") == "admin":
        return ws
    if ws.get("user_id") == user.get("id"):
        return ws
    # Shared workspace (user_id=None) — hamma ko'radi, lekin oddiy user
    # tahrirlay olmaydi (faqat o'qish va chat qilish)
    if ws.get("user_id") is None:
        return ws
    raise HTTPException(status_code=403, detail="Bu chat sizga tegishli emas")


def _public_url(request: Request, path: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}{path}"


def _decorate(ws: dict[str, Any], request: Request) -> dict[str, Any]:
    """Workspace ma'lumotini API endpoint URLlari bilan boyitish."""
    out = dict(ws)
    out["endpoints"] = {
        "ask": _public_url(request, f"/api/workspaces/{ws['id']}/ask"),
        "ask_stream": _public_url(request, f"/api/workspaces/{ws['id']}/ask/stream"),
        "upload": _public_url(request, f"/api/workspaces/{ws['id']}/upload"),
    }
    out["curl_example"] = (
        f'curl -X POST {out["endpoints"]["ask"]} \\\n'
        f'  -H "Content-Type: application/json" \\\n'
        f'  -H "X-API-Key: {ws["api_key"]}" \\\n'
        f"  -d '{{\"question\": \"Shartnoma summasi qancha?\"}}'"
    )
    return out


# ---------- CRUD ----------

@router.get("")
async def list_workspaces(request: Request, user: dict = Depends(current_user)):
    """Workspace ro'yxati.

    - Shared workspaces (user_id=None) — hammaga ochiq
    - Shaxsiy workspaces — faqat egasi (yoki admin) koʻradi
    """
    all_items = workspaces.list()
    if user.get("role") == "admin":
        items = all_items
    else:
        uid = user.get("id")
        items = [
            w for w in all_items
            if w.get("user_id") is None or w.get("user_id") == uid
        ]
    # Decorate va shared belgisini qoʻshish
    result = []
    for w in items:
        d = _decorate(w, request)
        d["is_shared"] = w.get("user_id") is None
        result.append(d)
    return {"items": result}


@router.post("")
async def create_workspace(
    req: WorkspaceCreate, request: Request, user: dict = Depends(current_user)
):
    data = req.model_dump(exclude_none=True)
    data["user_id"] = user.get("id")
    ws = await workspaces.create(data)
    return _decorate(ws, request)


@router.get("/{wid}")
async def get_workspace(wid: str, request: Request, user: dict = Depends(current_user)):
    ws = _resolve(wid, None, user=user)
    return _decorate(ws, request)


@router.patch("/{wid}")
async def update_workspace(
    wid: str, req: WorkspaceUpdate, request: Request,
    user: dict = Depends(current_user),
):
    ws = _resolve(wid, None, user=user)
    # Shared workspace'ni faqat admin tahrirlay oladi
    if ws.get("user_id") is None and user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Asosiy chatni faqat admin tahrirlay oladi",
        )
    try:
        ws = await workspaces.update(wid, req.model_dump(exclude_none=True))
    except KeyError:
        raise HTTPException(status_code=404, detail="Workspace topilmadi")
    return _decorate(ws, request)


@router.delete("/{wid}")
async def delete_workspace(wid: str, user: dict = Depends(current_user)):
    ws = _resolve(wid, None, user=user)
    if ws.get("user_id") is None and user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Asosiy chatni faqat admin o'chira oladi",
        )
    await workspaces.delete(wid)
    return {"ok": True}


@router.post("/{wid}/regenerate_key")
async def regen_key(wid: str, user: dict = Depends(current_user)):
    _resolve(wid, None, user=user)
    try:
        key = await workspaces.regenerate_api_key(wid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Workspace topilmadi")
    return {"api_key": key}


@router.post("/{wid}/clear_docs")
async def clear_docs(wid: str, user: dict = Depends(current_user)):
    """Workspace'dagi barcha biriktirilgan fayllarni unutish (vektor bazadan o'chirmaydi)."""
    _resolve(wid, None, user=user)
    ws = await workspaces.clear_docs(wid)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace topilmadi")
    return {"ok": True}


# ---------- Per-workspace ask ----------

@router.post("/{wid}/ask")
async def ws_ask(
    wid: str,
    body: AskBody,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    user: dict | None = Depends(current_user_optional),
):
    ws = _resolve(wid, x_api_key, user=user)
    doc_ids_list = ws.get("doc_ids") or None
    scope = list(ws.get("scope") or [])

    # PRIORITET 1: Shu so'rov bilan fayl yuklangan — faqat shu fayl(lar)ga filter
    if body.active_doc_ids:
        doc_ids_list = body.active_doc_ids
        if "uploads" not in scope:
            scope.append("uploads")
    else:
        # PRIORITET 2: "bu file/shu hujjat" deganda — oxirgi yuklangan faylga
        last_doc = ws.get("last_doc_id")
        if last_doc and _wants_recent_doc(body.question):
            doc_ids_list = [last_doc]
            if "uploads" not in scope:
                scope.append("uploads")

    # ❗ MUHIM: workspace'da fayl biriktirilgan bo'lsa — har doim "uploads" qo'shamiz
    # (chat scope ["reports"] bo'lsa ham, attach qilingan fayllar qidirilsin)
    if doc_ids_list and "uploads" not in scope:
        scope.append("uploads")

    rag_user_id = None
    if user and user.get("role") != "admin":
        rag_user_id = user.get("id")

    # 1) User xabarini DB'ga saqlash
    await msgs.save_message(wid, "user", body.question)

    # 2) Tarixni DB'dan olish (frontend yuborgan history'ni ham qabul qilamiz)
    history = (
        [h.model_dump() for h in body.history]
        if body.history
        else await msgs.recent_history(wid, max_turns=6)
    )
    # Joriy savolni tarixdan chiqarib tashlash (oxirgi user xabar — bu o'zi)
    history = [h for h in history if h.get("text") != body.question][:-0 if False else None]
    if history and history[-1].get("text") == body.question:
        history = history[:-1]

    t0 = time.monotonic()
    result = await rag.ask(
        question=body.question,
        scope=scope,
        doc_ids=doc_ids_list,
        system_prompt=ws.get("system_prompt") or None,
        allow_fallback=ws.get("allow_fallback", True),
        history=history,
        use_cache=False,  # Tarix bor, kesh oʻzgaruvchan
        user_id=rag_user_id,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # 3) Bot javobini DB'ga saqlash
    saved = await msgs.save_message(
        wid, "bot",
        text=result.get("answer", ""),
        mode=result.get("mode"),
        sources=result.get("sources") or [],
        elapsed_ms=elapsed_ms,
    )
    result["elapsed_ms"] = elapsed_ms
    result["message_id"] = saved["id"]
    return result


@router.post("/{wid}/ask/stream")
async def ws_ask_stream(
    wid: str,
    body: AskBody,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    user: dict | None = Depends(current_user_optional),
):
    ws = _resolve(wid, x_api_key, user=user)
    doc_ids_list = ws.get("doc_ids") or None
    scope = list(ws.get("scope") or [])

    # PRIORITET 1: Shu so'rov bilan fayl yuklangan
    if body.active_doc_ids:
        doc_ids_list = body.active_doc_ids
        if "uploads" not in scope:
            scope.append("uploads")
    else:
        # PRIORITET 2: "bu file/shu hujjat" deganda
        last_doc = ws.get("last_doc_id")
        if last_doc and _wants_recent_doc(body.question):
            doc_ids_list = [last_doc]
            if "uploads" not in scope:
                scope.append("uploads")

    # ❗ Workspace'da fayl biriktirilgan bo'lsa — har doim "uploads" qo'shamiz
    if doc_ids_list and "uploads" not in scope:
        scope.append("uploads")

    rag_user_id = None
    if user and user.get("role") != "admin":
        rag_user_id = user.get("id")

    # 1) User xabarini DB'ga saqlash
    await msgs.save_message(wid, "user", body.question)

    # SMART: "tekstini ber" / "matnini ko'rsat" — LLM ga bormay, to'liq matnni qaytaramiz
    if _wants_full_text(body.question):
        target_doc = None
        if body.active_doc_ids:
            target_doc = body.active_doc_ids[0]
        elif ws.get("last_doc_id"):
            target_doc = ws["last_doc_id"]
        if target_doc:
            full = await _full_text_of(target_doc, settings.QDRANT_COLLECTION_UPLOADS)
            if full:
                async def gen_text():
                    yield f"data: {json.dumps({'type':'mode','mode':'rag'}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type':'sources','sources':[]}, ensure_ascii=False)}\n\n"
                    # Matnni 100 belgilik bo'lakda streaming
                    for i in range(0, len(full), 100):
                        chunk = full[i:i+100]
                        yield f"data: {json.dumps({'type':'token','text':chunk}, ensure_ascii=False)}\n\n"
                    # DB'ga saqlash
                    saved = await msgs.save_message(
                        wid, "bot", text=full, mode="rag",
                        sources=[], elapsed_ms=0,
                    )
                    yield f"data: {json.dumps({'type':'saved','message_id':saved['id'],'elapsed_ms':0}, ensure_ascii=False)}\n\n"
                return StreamingResponse(gen_text(), media_type="text/event-stream")

    # 2) Tarixni DB'dan olish (joriy savol qoʻshilgan, uni chiqarib tashlaymiz)
    history = (
        [h.model_dump() for h in body.history]
        if body.history
        else await msgs.recent_history(wid, max_turns=6)
    )
    if history and history[-1].get("text") == body.question and history[-1].get("role") == "user":
        history = history[:-1]

    async def gen():
        t0 = time.monotonic()
        text_parts: list[str] = []
        sources: list[Any] = []
        mode: str | None = None
        first_token_at: float | None = None
        token_count = 0
        try:
            async for chunk in rag.ask_stream(
                question=body.question,
                scope=scope,
                doc_ids=doc_ids_list,
                system_prompt=ws.get("system_prompt") or None,
                allow_fallback=ws.get("allow_fallback", True),
                history=history,
                user_id=rag_user_id,
            ):
                if chunk.get("type") == "mode":
                    mode = chunk.get("mode")
                elif chunk.get("type") == "sources":
                    sources = chunk.get("sources") or []
                elif chunk.get("type") == "token":
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    token_count += 1
                    text_parts.append(chunk.get("text", ""))
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # 3) Stream tugadi — bot javobini DB'ga saqlash
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            gen_sec = (time.monotonic() - first_token_at) if first_token_at else 0
            tps = (token_count / gen_sec) if gen_sec > 0 else None
            final_text = "".join(text_parts).strip()
            saved = await msgs.save_message(
                wid, "bot",
                text=final_text,
                mode=mode,
                sources=sources,
                elapsed_ms=elapsed_ms,
                tokens_per_sec=tps,
                eval_count=token_count,
            )
            # Final meta event — frontend uchun (javob matni ham birga,
            # gpt-oss stream'da bo'sh kelsa, frontend buni ishlatadi)
            meta = {
                "type": "saved",
                "message_id": saved["id"],
                "elapsed_ms": elapsed_ms,
                "tokens_per_sec": tps,
                "token_count": token_count,
                "final_text": final_text,
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- Per-workspace upload ----------

ALLOWED_EXT = {
    ".pdf", ".docx", ".xlsx", ".txt", ".md",
    # OCR uchun rasm formatlari (Tesseract bilan)
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
}


@router.post("/{wid}/upload")
async def ws_upload(
    wid: str,
    file: UploadFile = File(...),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    user: dict | None = Depends(current_user_optional),
):
    ws = _resolve(wid, x_api_key, user=user)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Format qoʻllab-quvvatlanmaydi: {ext}")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(settings.UPLOAD_DIR, file.filename)
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    try:
        # Hujjatga egasi va workspace yozib qo'yiladi
        uid = user.get("id") if user else None
        result = await ingest_file(
            dest,
            settings.QDRANT_COLLECTION_UPLOADS,
            user_id=uid,
            workspace_id=wid,
        )
        if result.get("doc_id"):
            await workspaces.attach_doc(
                wid, result["doc_id"], filename=result.get("filename")
            )
        # Attach xabarini DB'ga saqlash + sources'ga doc_id (chat'da preview)
        await msgs.save_message(
            wid, "attach",
            text=f"{result.get('filename', file.filename)} indekslandi ({result.get('chunks', 0)} chunk)",
            sources=[{
                "doc_id": result.get("doc_id"),
                "filename": result.get("filename", file.filename),
                "size_bytes": result.get("size_bytes", 0),
                "chunks": result.get("chunks", 0),
                "mime": (file.content_type or ""),
            }],
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest xatosi: {e}")


# ---------- ask_with_file: upload + ask bir requestda (tashqi API uchun) ----------

@router.post("/{wid}/ask_with_file")
async def ws_ask_with_file(
    wid: str,
    file: UploadFile = File(...),
    question: str = Form(...),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    user: dict | None = Depends(current_user_optional),
):
    """Tashqi tizimlar uchun qulay: bitta requestda fayl yuklash + savol berish.

    multipart/form-data: file=<fayl>, question=<savol matni>
    """
    ws = _resolve(wid, x_api_key, user=user)

    # 1) Faylni yuklash
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Format qoʻllab-quvvatlanmaydi: {ext}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(settings.UPLOAD_DIR, file.filename)
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    try:
        uid = user.get("id") if user else None
        upload_result = await ingest_file(
            dest, settings.QDRANT_COLLECTION_UPLOADS,
            user_id=uid, workspace_id=wid,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest xato: {e}")

    if upload_result.get("doc_id"):
        await workspaces.attach_doc(
            wid, upload_result["doc_id"], filename=upload_result.get("filename")
        )

    # 2) Yuklangan faylga filter qilib savol berish
    await msgs.save_message(wid, "user", question)
    rag_user_id = None
    if user and user.get("role") != "admin":
        rag_user_id = user.get("id")

    t0 = time.monotonic()
    result = await rag.ask(
        question=question,
        scope=["uploads"],
        doc_ids=[upload_result["doc_id"]],
        system_prompt=ws.get("system_prompt") or None,
        allow_fallback=ws.get("allow_fallback", True),
        history=await msgs.recent_history(wid, max_turns=4),
        use_cache=False,
        user_id=rag_user_id,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    saved = await msgs.save_message(
        wid, "bot",
        text=result.get("answer", ""),
        mode=result.get("mode"),
        sources=result.get("sources") or [],
        elapsed_ms=elapsed_ms,
    )
    return {
        "uploaded": upload_result,
        "answer": result.get("answer", ""),
        "mode": result.get("mode"),
        "sources": result.get("sources") or [],
        "elapsed_ms": elapsed_ms,
        "message_id": saved["id"],
    }


# ---------- Messages CRUD ----------

@router.get("/{wid}/messages")
async def list_messages(
    wid: str, limit: int = 200, user: dict = Depends(current_user)
):
    _resolve(wid, None, user=user)
    return {"items": await msgs.list_messages(wid, limit=limit)}


@router.delete("/{wid}/messages")
async def clear_messages(wid: str, user: dict = Depends(current_user)):
    _resolve(wid, None, user=user)
    deleted = await msgs.clear_messages(wid)
    return {"ok": True, "deleted": deleted}


@router.delete("/{wid}/messages/{mid}")
async def delete_message(wid: str, mid: int, user: dict = Depends(current_user)):
    _resolve(wid, None, user=user)
    ok = await msgs.delete_message(mid)
    if not ok:
        raise HTTPException(status_code=404, detail="Xabar topilmadi")
    return {"ok": True}


class RateBody(BaseModel):
    # 1 = like, -1 = dislike, null = bahoni olib tashlash
    rating: Optional[int] = None


@router.post("/{wid}/messages/{mid}/rate")
async def rate_message(
    wid: str, mid: int, body: RateBody, user: dict = Depends(current_user)
):
    """Bot javobini baholash — like'lar fine-tuning dataset uchun yigʻiladi."""
    _resolve(wid, None, user=user)
    if body.rating not in (1, -1, None):
        raise HTTPException(status_code=400, detail="rating: 1, -1 yoki null")
    result = await msgs.rate_message(mid, body.rating, rated_by=user.get("id"))
    if not result:
        raise HTTPException(status_code=404, detail="Bot xabari topilmadi")
    return result


# ---------- Eksport (PDF / Word) ----------

@router.get("/{wid}/export")
async def export_workspace(
    wid: str,
    format: str = "pdf",
    user: dict = Depends(current_user),
):
    ws = _resolve(wid, None, user=user)
    fmt = format.lower()
    if fmt not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="format: 'pdf' yoki 'docx'")

    items = await msgs.list_messages(wid, limit=10000)
    if not items:
        raise HTTPException(status_code=400, detail="Chat boʻsh, eksport qilish uchun xabar yoʻq")

    # Fayl nomi
    safe_name = "".join(c for c in (ws.get("name") or "chat") if c.isalnum() or c in " _-")[:50].strip() or "chat"
    today = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{safe_name}_{today}.{fmt}"

    if fmt == "pdf":
        try:
            data = export_pdf(ws, items, user)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF eksport xatosi: {e}")
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    try:
        data = export_docx(ws, items, user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX eksport xatosi: {e}")
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
