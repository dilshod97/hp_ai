"""Auth: bcrypt parol + JWT token.

Dependency'lar:
- current_user(): JWT'dan user'ni oladi (har queryga himoya)
- require_admin(): faqat admin role uchun
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select

from app.config import settings
from app.core.db import SessionLocal, User

log = logging.getLogger("auth")

ALGORITHM = "HS256"


# ---------- Parol ----------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ---------- JWT ----------

def create_token(user_id: int, username: str, role: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=settings.JWT_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": exp,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError as e:
        log.warning("JWT decode xato: %s", e)
        return None


# ---------- Dependencies ----------

async def _get_user_from_request(request: Request) -> Optional[dict]:
    auth = request.headers.get("Authorization", "")
    token: str | None = None
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        # Cookie ham qabul qilamiz
        token = request.cookies.get("hp_token")
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError):
        return None
    async with SessionLocal() as s:
        user = await s.get(User, user_id)
        if not user or not user.is_active:
            return None
        return user.to_dict()


async def current_user(request: Request) -> dict:
    user = await _get_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Tizimga kiring")
    return user


async def current_user_optional(request: Request) -> dict | None:
    return await _get_user_from_request(request)


async def require_admin(user: dict = Depends(current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Faqat admin uchun")
    return user


# ---------- Initial bootstrap ----------

async def ensure_admin() -> None:
    """Birinchi marta — admin akkauntini yaratish (agar mavjud bo'lmasa)."""
    async with SessionLocal() as s:
        q = select(User).where(User.username == settings.ADMIN_USERNAME)
        existing = (await s.execute(q)).scalar_one_or_none()
        if existing:
            return
        admin = User(
            username=settings.ADMIN_USERNAME,
            full_name="Administrator",
            sector="Boshqaruv",
            password_hash=hash_password(settings.ADMIN_PASSWORD),
            role="admin",
            is_active=True,
        )
        s.add(admin)
        await s.commit()
        log.info(
            "👤 Admin yaratildi: %s / %s (login qilgach parolni o'zgartiring!)",
            settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD,
        )
