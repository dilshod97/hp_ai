"""Login / logout / current user."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core import users as users_svc
from app.core.auth import create_token, current_user, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str
    password: str


class ChangePasswordBody(BaseModel):
    old_password: str
    new_password: str


@router.post("/login")
async def login(body: LoginBody):
    u = await users_svc.get_user_by_username(body.username)
    if not u or not u.is_active or not verify_password(body.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Login yoki parol noto'g'ri")
    token = create_token(u.id, u.username, u.role)
    return {
        "token": token,
        "user": u.to_dict(),
    }


@router.get("/me")
async def me(user: dict = Depends(current_user)):
    return user


@router.post("/change_password")
async def change_password(
    body: ChangePasswordBody, user: dict = Depends(current_user)
):
    db_user = await users_svc.get_user_by_username(user["username"])
    if not db_user or not verify_password(body.old_password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Eski parol noto'g'ri")
    await users_svc.update_user(user["id"], password=body.new_password)
    return {"ok": True}
