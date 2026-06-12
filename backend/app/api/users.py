"""User CRUD — admin uchun."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core import users as users_svc
from app.core.auth import require_admin

router = APIRouter(prefix="/api/users", tags=["users"])


class CreateUserBody(BaseModel):
    username: str
    password: str
    full_name: str | None = None
    sector: str | None = None
    role: str = "user"


class UpdateUserBody(BaseModel):
    full_name: str | None = None
    sector: str | None = None
    role: str | None = None
    is_active: bool | None = None
    password: str | None = None


@router.get("")
async def list_users(_: dict = Depends(require_admin)):
    return {"items": await users_svc.list_users()}


@router.post("")
async def create_user(body: CreateUserBody, _: dict = Depends(require_admin)):
    if body.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role: admin yoki user")
    try:
        return await users_svc.create_user(
            username=body.username, password=body.password,
            full_name=body.full_name, sector=body.sector, role=body.role,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/{uid}")
async def update_user(uid: int, body: UpdateUserBody, _: dict = Depends(require_admin)):
    fields = body.model_dump(exclude_none=True)
    u = await users_svc.update_user(uid, **fields)
    if not u:
        raise HTTPException(status_code=404, detail="Foydalanuvchi topilmadi")
    return u


@router.delete("/{uid}")
async def delete_user(uid: int, admin: dict = Depends(require_admin)):
    if uid == admin["id"]:
        raise HTTPException(status_code=400, detail="O'zingizni o'chira olmaysiz")
    ok = await users_svc.delete_user(uid)
    if not ok:
        raise HTTPException(status_code=404, detail="Topilmadi")
    return {"ok": True}
