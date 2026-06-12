"""Workspaces (chat seanslari) — har birining oʻz scope, system prompt va API endpointi bor.

Saqlash: /data/workspaces.json
Misol:
  - "Qonunlar bo'yicha savol-javob" (scope=laws)
  - "Audit hisobotlari tahlili" (scope=reports)
  - "Shartnoma N12 tekshirilishi" (scope=uploads, doc_id=..., custom system prompt)
"""
import asyncio
import json
import logging
import os
import secrets
import uuid
from typing import Any

from app.config import settings

log = logging.getLogger("workspaces")
WORKSPACES_PATH = os.path.join(settings.DATA_DIR, "workspaces.json")

# Standart preset shablonlar — yangi foydalanuvchi uchun
DEFAULT_PRESETS = [
    {
        "name": "Qonunlar bo'yicha savol-javob",
        "scope": ["laws"],
        "system_prompt": "Asosan qonunlar va normativ hujjatlar asosida javob bering. Imkon bo'lsa qonun moddasini ko'rsating.",
        "allow_fallback": True,  # umumiy savollarga ham javob bersin
        "icon": "⚖️",
    },
    {
        "name": "Audit hisobotlari tahlili",
        "scope": ["reports"],
        "system_prompt": "Oldingi audit hisobotlari asosida javob bering. Tendentsiyalarni va takroriy kamchiliklarni ajratib koʻrsating.",
        "allow_fallback": True,  # umumiy savollarga ham javob bersin
        "icon": "📊",
    },
    {
        "name": "Erkin chat",
        # Bo'sh scope = RAG yo'q, faqat LLM bilan suhbat (ChatGPT kabi).
        # Fayl yuklab savol berilsa — avtomatik shu fayldan qidiradi.
        "scope": [],
        "system_prompt": "",
        "allow_fallback": True,
        "icon": "💬",
    },
]


class WorkspaceStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(WORKSPACES_PATH):
            try:
                with open(WORKSPACES_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._data = raw.get("workspaces", {})
                log.info("Workspaces yuklandi: %d ta", len(self._data))
                return
            except Exception as e:
                log.error("Workspaces faylini o'qib bo'lmadi: %s", e)
        # Birinchi marta — presetlarni yaratamiz
        for preset in DEFAULT_PRESETS:
            self._create_internal(preset)
        log.info("Standart workspacelar yaratildi: %d", len(self._data))

    async def _save(self) -> None:
        os.makedirs(os.path.dirname(WORKSPACES_PATH), exist_ok=True)
        tmp = WORKSPACES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"workspaces": self._data}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, WORKSPACES_PATH)

    def _create_internal(self, data: dict[str, Any]) -> dict[str, Any]:
        wid = str(uuid.uuid4())[:8]
        api_key = "hp_" + secrets.token_urlsafe(16)
        ws = {
            "id": wid,
            "user_id": data.get("user_id"),  # qaysi foydalanuvchi yaratdi
            "name": data.get("name", "Yangi chat"),
            "icon": data.get("icon", "💬"),
            "scope": data.get("scope", ["laws", "reports"]),
            "system_prompt": data.get("system_prompt", ""),
            "allow_fallback": bool(data.get("allow_fallback", True)),
            "doc_ids": data.get("doc_ids", []),  # Faqat shu hujjat(lar)dan qidirish
            "api_key": api_key,
            "created_at": None,
        }
        self._data[wid] = ws
        return ws

    def list(self, user_id: int | None = None) -> list[dict[str, Any]]:
        items = list(self._data.values())
        if user_id is not None:
            items = [w for w in items if w.get("user_id") == user_id]
        return items

    def get(self, wid: str) -> dict[str, Any] | None:
        return self._data.get(wid)

    def get_by_api_key(self, key: str) -> dict[str, Any] | None:
        for ws in self._data.values():
            if ws.get("api_key") == key:
                return ws
        return None

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            ws = self._create_internal(data)
            await self._save()
            return ws

    async def update(self, wid: str, updates: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if wid not in self._data:
                raise KeyError(wid)
            ws = self._data[wid]
            for k in ("name", "icon", "scope", "system_prompt", "allow_fallback", "doc_ids"):
                if k in updates:
                    ws[k] = updates[k]
            await self._save()
            return ws

    async def delete(self, wid: str) -> bool:
        async with self._lock:
            if wid not in self._data:
                return False
            del self._data[wid]
            await self._save()
            return True

    async def attach_doc(
        self, wid: str, doc_id: str, filename: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            ws = self._data.get(wid)
            if not ws:
                raise KeyError(wid)
            if doc_id not in ws["doc_ids"]:
                ws["doc_ids"].append(doc_id)
            # Oxirgi yuklangan fayl — "bu file" turidagi savollar uchun
            from datetime import datetime
            ws["last_doc_id"] = doc_id
            ws["last_doc_name"] = filename
            ws["last_doc_at"] = datetime.utcnow().isoformat()
            # Yuklangan fayl bo'lsa — scope ga uploads qo'shamiz
            if "uploads" not in ws["scope"]:
                ws["scope"] = list(ws["scope"]) + ["uploads"]
            await self._save()
            return ws

    async def clear_docs(self, wid: str) -> dict[str, Any] | None:
        """Workspace'dagi barcha doc_ids ni tozalash (eski fayllarni unutish)."""
        async with self._lock:
            ws = self._data.get(wid)
            if not ws:
                return None
            ws["doc_ids"] = []
            ws["last_doc_id"] = None
            ws["last_doc_name"] = None
            ws["last_doc_at"] = None
            await self._save()
            return ws

    async def regenerate_api_key(self, wid: str) -> str:
        async with self._lock:
            ws = self._data.get(wid)
            if not ws:
                raise KeyError(wid)
            ws["api_key"] = "hp_" + secrets.token_urlsafe(16)
            await self._save()
            return ws["api_key"]


workspaces = WorkspaceStore()
