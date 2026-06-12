"""Ollama orqali embedding olish (model runtime'da almashtirilishi mumkin).

Xavfsizlik qatlamlari:
- Uzun matnlar avtomatik qisqartiriladi (bge-m3 NaN qaytarishini oldini olish)
- Bo'sh javoblar o'tkazib yuboriladi
- Parallel concurrency (8 ta) — tezlik uchun
"""
import asyncio
import logging
import math

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.runtime_config import runtime_config

log = logging.getLogger("embeddings")

# Concurrency past — Ollama bge-m3 parallel sorovlarda NaN qaytarishi mumkin
EMBED_CONCURRENCY = 3
# bge-m3 maksimal kontekst — 8192 token, lekin xavfsizlik uchun belgida limit
# (~1 token = 2-3 belgi oʻzbek/kirill matnlarida)
MAX_CHARS_PER_EMBED = 6000


class Embedder:
    def __init__(self) -> None:
        self.url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
        self.client = httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self._sem = asyncio.Semaphore(EMBED_CONCURRENCY)
        self._zero_dim: int | None = None  # bo'sh vector dim — fallback uchun

    @property
    def model(self) -> str:
        return runtime_config.get("embedding_model", settings.EMBEDDING_MODEL)

    def _safe_truncate(self, text: str) -> str:
        """Uzun matnni xavfsiz darajada qisqartirish."""
        if len(text) <= MAX_CHARS_PER_EMBED:
            return text
        # Soʻz chegarasida kesish
        cut = text[:MAX_CHARS_PER_EMBED]
        last_space = cut.rfind(" ")
        if last_space > MAX_CHARS_PER_EMBED * 0.8:
            cut = cut[:last_space]
        return cut

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def _embed_raw(self, text: str) -> list[float]:
        async with self._sem:
            resp = await self.client.post(
                self.url,
                json={"model": self.model, "prompt": text, "keep_alive": "24h"},
            )
            resp.raise_for_status()
            data = resp.json()
            vec = data.get("embedding", [])
            # NaN/Inf tekshirish — Ollama ba'zan buzilgan vektor qaytaradi
            if any(math.isnan(v) or math.isinf(v) for v in vec):
                raise ValueError("Embedding'da NaN/Inf qiymat")
            return vec

    async def embed(self, text: str) -> list[float]:
        """Xavfsiz embedding — uzun matnni qisqartiradi, xato bo'lsa nol vektor."""
        text = (text or "").strip()
        if not text:
            return self._zero_vector()
        text = self._safe_truncate(text)
        try:
            vec = await self._embed_raw(text)
            self._zero_dim = len(vec)
            return vec
        except Exception as e:
            log.warning("Embed xato (chunk o'tkazib yuborildi): %s — matn[:60]=%r",
                        e, text[:60])
            return self._zero_vector()

    def _zero_vector(self) -> list[float]:
        """Xato yoki bo'sh matn uchun — barqaror nol vektor qaytarish."""
        if self._zero_dim is None:
            self._zero_dim = 1024  # bge-m3 default
        return [0.0] * self._zero_dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Parallel embedding — 8 ta so'rov bir vaqtda."""
        if not texts:
            return []
        return await asyncio.gather(*[self.embed(t) for t in texts])

    async def detect_dim(self) -> int:
        v = await self._embed_raw("test")
        self._zero_dim = len(v)
        return len(v)

    async def close(self) -> None:
        await self.client.aclose()


embedder = Embedder()
