"""Qdrant bilan ishlash uchun yengil wrapper.

Avtomatik dimension matching:
- Embedding modelni almashtirsangiz, vektor o'lchami farqlanishi mumkin
  (bge-m3=1024, qwen3-embedding=4096, nomic=768).
- Collection mavjud, lekin oʻlcham mos emas — avtomatik qayta yaratiladi
  (eski ma'lumotlar yoʻqoladi, qayta ingest kerak).
"""
import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.config import settings

log = logging.getLogger("vector_store")

# Standart fallback (agar dimensionni aniqlay olmasa)
DEFAULT_VECTOR_SIZE = 1024


class VectorStore:
    def __init__(self) -> None:
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL)
        # Joriy ishlatilayotgan oʻlcham (runtime'da aniqlanadi)
        self._current_dim: int | None = None

    def set_current_dim(self, dim: int) -> None:
        self._current_dim = dim

    async def get_collection_dim(self, name: str) -> int | None:
        """Mavjud collection'ning vektor oʻlchamini qaytaradi (yo'q bo'lsa None)."""
        try:
            info = await self.client.get_collection(name)
            # vectors_config — nomli yoki anonim bo'lishi mumkin
            cfg = info.config.params.vectors
            if hasattr(cfg, "size"):
                return cfg.size
            # nomli vektorlar — birinchisini olamiz
            if isinstance(cfg, dict) and cfg:
                first = next(iter(cfg.values()))
                return first.size
        except Exception:
            return None
        return None

    async def _create_collection(self, name: str, dim: int) -> None:
        await self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        for field in ("doc_id", "doc_type", "year", "region"):
            try:
                await self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD
                    if field != "year"
                    else qm.PayloadSchemaType.INTEGER,
                )
            except Exception:
                pass
        log.info("Collection yaratildi: %s (dim=%d)", name, dim)

    async def ensure_collection(self, name: str, dim: int | None = None) -> None:
        """Collection mavjud va kerakli oʻlchamga ega ekanligini taʼminlaydi."""
        dim = dim or self._current_dim or DEFAULT_VECTOR_SIZE
        exists = await self.client.collection_exists(name)
        if not exists:
            await self._create_collection(name, dim)
            return

        current_dim = await self.get_collection_dim(name)
        if current_dim != dim:
            log.warning(
                "Collection '%s' oʻlchami mos kelmaydi: bor=%s, kerak=%d. Qayta yaratilmoqda.",
                name, current_dim, dim,
            )
            await self.client.delete_collection(name)
            await self._create_collection(name, dim)

    async def upsert(
        self,
        collection: str,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> list[str]:
        if vectors:
            await self.ensure_collection(collection, dim=len(vectors[0]))
        ids = [str(uuid.uuid4()) for _ in vectors]
        points = [
            qm.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(vectors))
        ]
        await self.client.upsert(collection_name=collection, points=points)
        return ids

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 4,
        filter_: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        await self.ensure_collection(collection, dim=len(vector))
        qfilter = None
        if filter_:
            must = []
            for k, v in filter_.items():
                if isinstance(v, list):
                    if not v:
                        continue
                    must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
                else:
                    must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
            qfilter = qm.Filter(must=must) if must else None

        result = await self.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [
            {"score": p.score, "payload": p.payload, "id": p.id}
            for p in result
        ]

    async def delete_collection(self, name: str) -> bool:
        try:
            await self.client.delete_collection(name)
            log.info("Collection o'chirildi: %s", name)
            return True
        except Exception as e:
            log.error("O'chirish xatosi (%s): %s", name, e)
            return False

    async def delete_all(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for col in (
            settings.QDRANT_COLLECTION_LAWS,
            settings.QDRANT_COLLECTION_REPORTS,
            settings.QDRANT_COLLECTION_UPLOADS,
        ):
            result[col] = await self.delete_collection(col)
        return result

    async def delete_by_doc(self, collection: str, doc_id: str) -> None:
        await self.client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="doc_id",
                            match=qm.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )

    async def close(self) -> None:
        await self.client.close()


vector_store = VectorStore()
