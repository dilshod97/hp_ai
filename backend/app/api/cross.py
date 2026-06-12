"""Cross-user admin API — barcha foydalanuvchilar hujjatlaridan qidirish.

Admin uchun. Tashqi tizimga ham ulanish mumkin (API key bilan).
"""
import json
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.core import rag, users as users_svc
from app.core.auth import current_user_optional, require_admin
from app.core.runtime_config import runtime_config

router = APIRouter(prefix="/api/cross", tags=["cross-user"])


class CrossAskBody(BaseModel):
    question: str
    scope: Optional[list[str]] = None
    user_ids: Optional[list[int]] = None  # ko'rsatilsa, faqat shu userlar
    history: Optional[list[dict]] = None
    allow_fallback: bool = True


# Cross-user API uchun maxsus master key (.env yoki runtime)
def _get_master_key() -> str:
    key = runtime_config.get("cross_master_key")
    if not key:
        key = "hpcross_" + secrets.token_urlsafe(24)
        # Ko'rsatish uchun loglash
        import logging
        logging.getLogger("cross").info("🔑 Cross-user master key: %s", key)
        # Saqlash imkoni — runtime_config'ga emas, alohida saqlash kerak
        # Hozircha har restartda yangi yaratiladi (admin UI'da koʻrsatamiz)
        runtime_config._data["cross_master_key"] = key
    return key


def _check_access(api_key: str | None, user: dict | None):
    if api_key and api_key == _get_master_key():
        return
    if user and user.get("role") == "admin":
        return
    raise HTTPException(status_code=403, detail="Faqat admin yoki master key")


@router.get("/info")
async def info(_: dict = Depends(require_admin)):
    """Master key va foydalanuvchilar roʻyxatini qaytaradi."""
    all_users = await users_svc.list_users()
    return {
        "master_key": _get_master_key(),
        "endpoint_ask": "/api/cross/ask",
        "endpoint_stream": "/api/cross/ask/stream",
        "users": [
            {"id": u["id"], "username": u["username"],
             "full_name": u["full_name"], "sector": u["sector"]}
            for u in all_users if u["role"] != "admin"
        ],
        "curl_example": (
            f"curl -X POST http://localhost:8000/api/cross/ask \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -H 'X-Master-Key: {_get_master_key()}' \\\n"
            f"  -d '{{\"question\": \"Qaysi bankda eng ko'p kamchilik?\"}}'"
        ),
    }


@router.post("/ask")
async def cross_ask(
    body: CrossAskBody,
    x_master_key: str | None = Header(None, alias="X-Master-Key"),
    user: dict | None = Depends(current_user_optional),
):
    _check_access(x_master_key, user)
    # user_id=None — cheklov yo'q, hammadan qidiradi
    # Lekin foydalanuvchilar ro'yxati cheklangan bo'lsa, doc_id filter qoʻyamiz
    # Hozircha barchasidan qidiramiz (top_k boʻyicha eng yaxshilar olinadi)
    result = await rag.ask(
        question=body.question,
        scope=body.scope or ["laws", "reports", "uploads"],
        allow_fallback=body.allow_fallback,
        history=body.history,
        user_id=None,  # cheklov yoʻq
        use_cache=False,
    )
    result["aggregated"] = True
    return result


@router.post("/ask/stream")
async def cross_ask_stream(
    body: CrossAskBody,
    x_master_key: str | None = Header(None, alias="X-Master-Key"),
    user: dict | None = Depends(current_user_optional),
):
    _check_access(x_master_key, user)

    async def gen():
        try:
            async for chunk in rag.ask_stream(
                question=body.question,
                scope=body.scope or ["laws", "reports", "uploads"],
                allow_fallback=body.allow_fallback,
                history=body.history,
                user_id=None,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
