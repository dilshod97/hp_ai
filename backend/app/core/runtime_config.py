"""Runtime config — UI'dan o'zgartirilishi mumkin bo'lgan sozlamalar.

Sozlamalar JSON faylga saqlanadi (/data/config.json) — qayta startda saqlanib qoladi.
Konteyner ichida bo'lsa ham, /data volume orqali host'ga mount qilingan.
"""
import asyncio
import json
import logging
import os
from typing import Any

from app.config import settings

log = logging.getLogger("runtime_config")

CONFIG_PATH = os.path.join(settings.DATA_DIR, "config.json")

# Standart qiymatlar — agar config.json yo'q bo'lsa, .env dan olinadi
DEFAULTS: dict[str, Any] = {
    "llm_model": settings.LLM_MODEL,
    "embedding_model": settings.EMBEDDING_MODEL,
    "llm_temperature": settings.LLM_TEMPERATURE,
    # gpt-oss reasoning model — thinking + response uchun budjet kerak.
    # 1024 kam, 2500+ tavsiya.
    "llm_max_tokens": max(settings.LLM_MAX_TOKENS, 2500),
    "top_k": settings.TOP_K,
    "chunk_size": settings.CHUNK_SIZE,
    "chunk_overlap": settings.CHUNK_OVERLAP,
    "reasoning": "medium",  # low / medium / high
    "use_cache": True,
    "cache_similarity_threshold": settings.CACHE_SIMILARITY_THRESHOLD,
}

# O'zgartirilishi mumkin maydonlar va validatsiya
EDITABLE_FIELDS: dict[str, type] = {
    "llm_model": str,
    "embedding_model": str,
    "llm_temperature": float,
    "llm_max_tokens": int,
    "top_k": int,
    "chunk_size": int,
    "chunk_overlap": int,
    "reasoning": str,
    "use_cache": bool,
    "cache_similarity_threshold": float,
}


class RuntimeConfig:
    """In-memory + disk persistence config."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._data: dict[str, Any] = dict(DEFAULTS)
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not os.path.exists(CONFIG_PATH):
            log.info("Config fayli yo'q, standart qiymatlar ishlatiladi: %s", CONFIG_PATH)
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                disk = json.load(f)
            for k, v in disk.items():
                if k in DEFAULTS:
                    self._data[k] = v
            log.info("Config yuklandi: %s", CONFIG_PATH)
        except Exception as e:
            log.error("Config faylini o'qib bo'lmadi: %s", e)

    async def _save_to_disk(self) -> None:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
        log.info("Config saqlandi: %s", CONFIG_PATH)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def all(self) -> dict[str, Any]:
        return dict(self._data)

    async def update(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Bir nechta maydonni yangilash. Validatsiyadan keyin diskga yoziladi."""
        async with self._lock:
            changed: dict[str, Any] = {}
            for k, v in updates.items():
                if k not in EDITABLE_FIELDS:
                    raise ValueError(f"Noma'lum maydon: {k}")
                expected_type = EDITABLE_FIELDS[k]
                # bool maxsus holat (int bo'lib kelmasligi uchun)
                if expected_type is bool and not isinstance(v, bool):
                    raise ValueError(f"{k} bool bo'lishi kerak")
                try:
                    casted = expected_type(v) if not isinstance(v, expected_type) else v
                except (ValueError, TypeError):
                    raise ValueError(f"{k} uchun noto'g'ri qiymat: {v}")
                # Maxsus tekshiruvlar
                if k == "reasoning" and casted not in ("low", "medium", "high"):
                    raise ValueError("reasoning: low/medium/high bo'lishi kerak")
                if k == "llm_temperature" and not (0.0 <= casted <= 2.0):
                    raise ValueError("llm_temperature: 0.0 — 2.0 oraliqda")
                if k == "top_k" and not (1 <= casted <= 20):
                    raise ValueError("top_k: 1 — 20 oraliqda")
                if k == "cache_similarity_threshold" and not (0.5 <= casted <= 1.0):
                    raise ValueError("cache_similarity_threshold: 0.5 — 1.0")
                self._data[k] = casted
                changed[k] = casted
            await self._save_to_disk()
            return changed

    async def reset(self) -> None:
        async with self._lock:
            self._data = dict(DEFAULTS)
            await self._save_to_disk()


runtime_config = RuntimeConfig()
