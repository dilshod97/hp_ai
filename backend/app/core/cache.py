"""Redis asosida semantik kesh.

Savol embedding qilinadi, oldingi savollar bilan cosine similarity solishtiriladi.
Agar threshold dan yuqori boʻlsa — keshlangan javob qaytariladi.
"""
import hashlib
import json
from typing import Optional

import numpy as np
import redis.asyncio as redis_async

from app.config import settings

CACHE_INDEX_KEY = "hp:cache:index"  # set of question_ids
CACHE_VEC_PREFIX = "hp:cache:vec:"   # hash: question_id -> base64 vector
CACHE_ANS_PREFIX = "hp:cache:ans:"   # string: question_id -> answer json


class SemanticCache:
    def __init__(self) -> None:
        self.client = redis_async.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=False
        )
        self.threshold = settings.CACHE_SIMILARITY_THRESHOLD
        self.ttl = settings.CACHE_TTL_SECONDS

    @staticmethod
    def _hash(question: str) -> str:
        return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    async def lookup(
        self, question: str, embedding: list[float]
    ) -> Optional[dict]:
        # Aniq mos kelish (tezroq yo'l)
        qid = self._hash(question)
        ans = await self.client.get(CACHE_ANS_PREFIX + qid)
        if ans:
            return json.loads(ans)

        # Semantik mos kelish
        ids = await self.client.smembers(CACHE_INDEX_KEY)
        if not ids:
            return None
        target = np.array(embedding, dtype=np.float32)
        best_id, best_sim = None, -1.0
        for raw_id in ids:
            sid = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            vec_raw = await self.client.get(CACHE_VEC_PREFIX + sid)
            if not vec_raw:
                continue
            vec = np.frombuffer(vec_raw, dtype=np.float32)
            sim = self._cosine(target, vec)
            if sim > best_sim:
                best_sim = sim
                best_id = sid
        if best_id and best_sim >= self.threshold:
            ans = await self.client.get(CACHE_ANS_PREFIX + best_id)
            if ans:
                obj = json.loads(ans)
                obj["_cache_similarity"] = best_sim
                return obj
        return None

    async def store(
        self, question: str, embedding: list[float], answer: dict
    ) -> None:
        qid = self._hash(question)
        vec = np.array(embedding, dtype=np.float32).tobytes()
        await self.client.set(CACHE_VEC_PREFIX + qid, vec, ex=self.ttl)
        await self.client.set(
            CACHE_ANS_PREFIX + qid,
            json.dumps(answer, ensure_ascii=False),
            ex=self.ttl,
        )
        await self.client.sadd(CACHE_INDEX_KEY, qid)

    async def close(self) -> None:
        await self.client.aclose()


cache = SemanticCache()
