"""Ikkita hujjatni solishtirish endpointi."""
import json

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.core.embeddings import embedder
from app.core.llm import llm
from app.core.vector_store import vector_store
from app.models.schemas import CompareRequest, CompareResponse

router = APIRouter(prefix="/api", tags=["compare"])


COMPARE_SYSTEM = (
    "Siz hujjatlar tahlilchisisiz. Ikkita hujjatning tarkibini taqqoslab, "
    "farqlarni aniq ko'rsating. Faqat berilgan kontekst asosida fikr bildiring. "
    "Javob JSON formatida bo'lsin: "
    "{ \"summary\": \"...\", \"differences\": [\"...\", \"...\"] }"
)


async def _fetch_doc_chunks(collection: str, doc_id: str, limit: int = 30) -> list[dict]:
    """Bitta hujjat boʻyicha barcha chunklarni scroll qilib olish."""
    points, _ = await vector_store.client.scroll(
        collection_name=collection,
        scroll_filter={
            "must": [{"key": "doc_id", "match": {"value": doc_id}}]
        },
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [{"payload": p.payload} for p in points]


def _join(chunks: list[dict], max_chars: int = 6000) -> str:
    parts = []
    total = 0
    for c in chunks:
        t = (c["payload"].get("text") or "").strip()
        if not t:
            continue
        if total + len(t) > max_chars:
            break
        parts.append(t)
        total += len(t)
    return "\n\n".join(parts)


@router.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    # Hozircha faqat uploads collection ichida qidiramiz
    col = settings.QDRANT_COLLECTION_UPLOADS
    try:
        a = await _fetch_doc_chunks(col, req.doc_id_a)
        b = await _fetch_doc_chunks(col, req.doc_id_b)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hujjatlarni olishda xato: {e}")

    if not a or not b:
        raise HTTPException(status_code=404, detail="Hujjatlardan biri topilmadi")

    a_text = _join(a)
    b_text = _join(b)
    a_name = a[0]["payload"].get("source", "Hujjat A")
    b_name = b[0]["payload"].get("source", "Hujjat B")

    aspects = (
        ", ".join(req.aspects)
        if req.aspects
        else "summasi, muddati, tomonlar, majburiyatlar, muhim shartlar"
    )

    prompt = (
        f"HUJJAT A ({a_name}):\n{a_text}\n\n"
        f"HUJJAT B ({b_name}):\n{b_text}\n\n"
        f"Quyidagi jihatlar bo'yicha taqqoslang: {aspects}.\n"
        "Faqat JSON qaytaring."
    )

    # Solishtirish — murakkab tahlil, shuning uchun reasoning=high
    # JSON mode — strukturlangan chiqishni kafolatlaydi
    raw = await llm.generate(
        prompt, system=COMPARE_SYSTEM, reasoning="high", json_mode=True
    )

    # JSON ni ajratib olish
    summary = raw.strip()
    differences: list[str] = []
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            summary = parsed.get("summary", summary)
            differences = parsed.get("differences", [])
        except Exception:
            pass

    sources = [
        {"n": 1, "source": a_name, "doc_id": req.doc_id_a, "score": 1.0, "section": None},
        {"n": 2, "source": b_name, "doc_id": req.doc_id_b, "score": 1.0, "section": None},
    ]
    return {"summary": summary, "differences": differences, "sources": sources}
