"""User CRUD."""
from sqlalchemy import select

from app.core.auth import hash_password
from app.core.db import SessionLocal, User


async def list_users() -> list[dict]:
    async with SessionLocal() as s:
        rows = (await s.execute(select(User).order_by(User.id.asc()))).scalars().all()
        return [r.to_dict() for r in rows]


async def get_user(user_id: int) -> dict | None:
    async with SessionLocal() as s:
        u = await s.get(User, user_id)
        return u.to_dict() if u else None


async def get_user_by_username(username: str) -> User | None:
    async with SessionLocal() as s:
        q = select(User).where(User.username == username)
        return (await s.execute(q)).scalar_one_or_none()


async def create_user(
    username: str,
    password: str,
    full_name: str | None = None,
    sector: str | None = None,
    role: str = "user",
) -> dict:
    async with SessionLocal() as s:
        # Mavjudligini tekshirish
        existing = await s.execute(select(User).where(User.username == username))
        if existing.scalar_one_or_none():
            raise ValueError(f"'{username}' allaqachon mavjud")
        u = User(
            username=username,
            full_name=full_name,
            sector=sector,
            password_hash=hash_password(password),
            role=role,
            is_active=True,
        )
        s.add(u)
        await s.commit()
        await s.refresh(u)
        return u.to_dict()


async def update_user(user_id: int, **fields) -> dict | None:
    async with SessionLocal() as s:
        u = await s.get(User, user_id)
        if not u:
            return None
        if "password" in fields and fields["password"]:
            u.password_hash = hash_password(fields.pop("password"))
        for k, v in fields.items():
            if k in ("full_name", "sector", "role", "is_active") and hasattr(u, k):
                setattr(u, k, v)
        await s.commit()
        await s.refresh(u)
        return u.to_dict()


async def delete_user(user_id: int) -> bool:
    async with SessionLocal() as s:
        u = await s.get(User, user_id)
        if not u:
            return False
        await s.delete(u)
        await s.commit()
        return True
