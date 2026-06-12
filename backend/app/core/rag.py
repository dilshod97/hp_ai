"""RAG: savol -> embedding -> qidiruv -> prompt -> LLM.

Rejimlar:
- strict (default off): kontekst yo'q bo'lsa, "javob topilmadi" qaytaradi
- fallback (default on): kontekst yo'q bo'lsa, LLM erkin javob beradi
"""
from typing import AsyncIterator

from app.config import settings
from app.core.cache import cache
from app.core.embeddings import embedder
from app.core.llm import llm
from app.core.runtime_config import runtime_config
from app.core.vector_store import vector_store

RAG_SYSTEM = (
    "Siz Hisob palatasi auditorlari uchun yordamchi sun'iy intellektsiz.\n"
    "\n"
    "🇺🇿 ENG MUHIM QOIDA: Javobni HAR DOIM OʻZBEK TILIDA (lotin alifbosida) bering!\n"
    "Hujjatlar rus tilida yoki kirill alifbosida boʻlsa ham — javob faqat oʻzbekcha lotinda.\n"
    "Kerakli atamalarni avval oʻzbekcha yozing, qavs ichida asl ruscha/kirilcha varianti.\n"
    "Masalan: 'tender' (тендер), 'shartnoma' (договор).\n"
    "\n"
    "Manbalar:\n"
    "- SUHBAT TARIXI: oldingi gaplashganlar\n"
    "- KONTEKST: hujjatlardan olingan parchalar (qonunlar, hisobotlar)\n"
    "\n"
    "Qoidalar:\n"
    "1. Suhbat tarixidagi shaxsiy/kontekstual maʼlumotni ishlatishingiz mumkin.\n"
    "2. Huquqiy yoki faktik savollarda KONTEKST asosida javob bering.\n"
    "3. Qonun moddalarini va raqamli maʼlumotlarni oʻzgartirmang.\n"
    "4. Kontekstda yoʻq boʻlsa — \"maʼlumot yoʻq\" deb ayting.\n"
    "5. Javob aniq, qisqa, ish uslubida boʻlsin.\n"
    "\n"
    "MUHIM: Javobda [1], [2] kabi manba raqamlarini KIRITMANG. "
    "\"manba\" yoki \"manbalar\" soʻzlarini yozmang. Tabiiy gap shaklida.\n"
    "Jadval ham 'Manba' ustunisiz boʻlsin."
)

MAX_HISTORY_TURNS = 6  # oxirgi N juftlik (user+bot)
MAX_HISTORY_CHARS = 2000  # tarix uchun belgilangan budjet


def _format_history(history: list[dict] | None) -> str:
    """Frontenddan kelgan tarixni promptga qo'shish uchun matn ko'rinishida tayyorlaydi.

    history elementi: {"role": "user"|"bot", "text": "..."}
    """
    if not history:
        return ""
    recent = [m for m in history if m.get("role") in ("user", "bot") and m.get("text")]
    recent = recent[-(MAX_HISTORY_TURNS * 2):]
    if not recent:
        return ""

    lines: list[str] = []
    total = 0
    # Oxiridan boshlab to'plab, keyin orqaga qarab joylashtiramiz
    for m in reversed(recent):
        role = "Foydalanuvchi" if m["role"] == "user" else "Yordamchi"
        text = m["text"].strip().replace("\n", " ")
        if len(text) > 500:
            text = text[:500] + "..."
        line = f"{role}: {text}"
        if total + len(line) > MAX_HISTORY_CHARS:
            break
        lines.append(line)
        total += len(line)
    lines.reverse()
    return "SUHBAT TARIXI:\n" + "\n".join(lines) + "\n"

FALLBACK_SYSTEM_TEMPLATE = (
    "Siz Hisob palatasi auditorlari uchun yordamchi sun'iy intellektsiz.\n"
    "{custom}\n"
    "\n"
    "🇺🇿 ENG MUHIM: HAR DOIM OʻZBEK TILIDA (lotin alifbosida) javob bering!\n"
    "Foydalanuvchi rus tilida yoki kirill alifbosida yozsa ham — javob faqat oʻzbekcha lotinda.\n"
    "\n"
    "Salomlashuv, oddiy savol va umumiy maslahat bersangiz boʻladi.\n"
    "Lekin huquqiy fakt yoki aniq qonun moddasi keltirilsa — 'hujjat yuklang' deb tavsiya bering."
)


def _build_prompt(
    question: str, hits: list[dict], history: list[dict] | None = None
) -> tuple[str, list[dict]]:
    sources: list[dict] = []
    parts: list[str] = []

    hist_text = _format_history(history)
    if hist_text:
        parts.append(hist_text)

    parts.append("KONTEKST (hujjatlardan parchalar):\n")
    for i, h in enumerate(hits, start=1):
        p = h["payload"]
        src_name = p.get("source", "noma'lum")
        # Faqat matnni beramiz — manba nomi hujjat ichida emas
        parts.append("--- Parcha ---")
        parts.append(p.get("text", "").strip())
        parts.append("")
        # Frontend uchun sources ro'yxati — lekin javobga ko'rsatilmaydi
        sources.append({
            "n": i,
            "source": src_name,
            "section": p.get("section") or p.get("page") or "",
            "doc_id": p.get("doc_id"),
            "score": h["score"],
        })

    parts.append("YANGI SAVOL:")
    parts.append(question.strip())
    parts.append(
        "\nJAVOB (tabiiy gap bilan, [1] kabi raqamlar va 'Manba' so'zlarini "
        "ishlatmang):"
    )
    return "\n".join(parts), sources


async def _gather_hits(
    query_vec: list[float],
    collections: list[str],
    top_k_each: int,
    filter_: dict | None = None,
    final_top_k: int | None = None,
) -> list[dict]:
    all_hits: list[dict] = []
    for col in collections:
        hits = await vector_store.search(col, query_vec, top_k=top_k_each, filter_=filter_)
        for h in hits:
            h["payload"]["_collection"] = col
        all_hits.extend(hits)
    all_hits.sort(key=lambda x: x["score"], reverse=True)
    final = final_top_k if final_top_k is not None else runtime_config.get("top_k", settings.TOP_K)
    return all_hits[:final]


def _resolve_collections(scope: list[str]) -> list[str]:
    mapping = {
        "laws": settings.QDRANT_COLLECTION_LAWS,
        "reports": settings.QDRANT_COLLECTION_REPORTS,
        "uploads": settings.QDRANT_COLLECTION_UPLOADS,
    }
    return [mapping[s] for s in scope if s in mapping]


def _fallback_system(custom_prompt: str | None) -> str:
    return FALLBACK_SYSTEM_TEMPLATE.format(custom=custom_prompt or "")


async def ask(
    question: str,
    scope: list[str] | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    use_cache: bool = True,
    system_prompt: str | None = None,
    allow_fallback: bool = True,
    history: list[dict] | None = None,
    user_id: int | None = None,
) -> dict:
    """Sinxron RAG javob.

    allow_fallback=True bo'lsa, manba topilmasa, LLM erkin javob beradi.
    """
    # Bo'sh scope — RAG umuman ishlatilmaydi (sof LLM suhbat — Erkin chat)
    if not scope:
        hist_text = _format_history(history)
        fb_prompt = f"{hist_text}\nYANGI SAVOL: {question}" if hist_text else question
        answer = await llm.generate(fb_prompt, system=_fallback_system(system_prompt))
        return {
            "answer": answer.strip(),
            "sources": [],
            "from_cache": False,
            "mode": "llm_only",
        }

    collections = _resolve_collections(scope)
    if not collections:
        # Scope berilgan, lekin yaroqsiz nomlar — bo'sh hisoblaymiz
        hist_text = _format_history(history)
        fb_prompt = f"{hist_text}\nYANGI SAVOL: {question}" if hist_text else question
        answer = await llm.generate(fb_prompt, system=_fallback_system(system_prompt))
        return {
            "answer": answer.strip(), "sources": [],
            "from_cache": False, "mode": "llm_only",
        }

    q_vec = await embedder.embed(question)

    # Tarix bor bo'lsa, cache'ni o'tkazib yuboramiz (kontekst doimo o'zgaradi)
    cacheable = use_cache and not history
    if cacheable:
        cached = await cache.lookup(question, q_vec)
        if cached:
            cached["from_cache"] = True
            return cached

    filter_: dict = {}
    # doc_ids list bo'lsa — MatchAny, doc_id bitta — MatchValue
    if doc_ids:
        filter_["doc_id"] = doc_ids
    elif doc_id:
        filter_["doc_id"] = doc_id
    if user_id is not None:
        filter_["user_id"] = user_id

    # Top-K ni dinamik: agar koʻp hujjat bor bo'lsa, qidiruvni kengaytirish
    base_top_k = runtime_config.get("top_k", settings.TOP_K)
    if doc_ids and len(doc_ids) > 1:
        # Har bir hujjatdan kamida 1 ta chunk olishga harakat
        effective_top_k = max(base_top_k, min(8, len(doc_ids) * 2))
    else:
        effective_top_k = base_top_k

    hits = await _gather_hits(
        q_vec, collections, top_k_each=effective_top_k,
        filter_=filter_ or None, final_top_k=effective_top_k,
    )

    # Similarity past bo'lsa (meta savol, kontekstda mos kelmaydi) — fallback
    LOW_SIMILARITY_THRESHOLD = 0.20
    if hits and (hits[0].get("score") or 0) < LOW_SIMILARITY_THRESHOLD:
        hits = []

    if not hits:
        if not allow_fallback:
            result = {
                "answer": "Berilgan hujjatlarda bu savolga javob topilmadi.",
                "sources": [],
                "from_cache": False,
                "mode": "no_context",
            }
            return result
        # Fallback — LLM erkin javob beradi, tarix bilan
        hist_text = _format_history(history)
        fb_prompt = f"{hist_text}\nYANGI SAVOL: {question}" if hist_text else question
        answer = await llm.generate(fb_prompt, system=_fallback_system(system_prompt))
        result = {
            "answer": answer.strip(),
            "sources": [],
            "from_cache": False,
            "mode": "llm_only",
        }
        if cacheable and (result.get("answer") or "").strip():
            await cache.store(question, q_vec, result)
        return result

    prompt, sources = _build_prompt(question, hits, history=history)
    sys_msg = RAG_SYSTEM
    if system_prompt:
        sys_msg = f"{RAG_SYSTEM}\n\nQo'shimcha:\n{system_prompt}"
    answer = await llm.generate(prompt, system=sys_msg)
    result = {
        "answer": answer.strip(),
        "sources": sources,
        "from_cache": False,
        "mode": "rag",
    }
    # Faqat haqiqiy javobni keshlash (bo'sh — model xatosini takrorlamaslik uchun)
    if cacheable and (result.get("answer") or "").strip():
        await cache.store(question, q_vec, result)
    return result


async def ask_stream(
    question: str,
    scope: list[str] | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    system_prompt: str | None = None,
    allow_fallback: bool = True,
    history: list[dict] | None = None,
    user_id: int | None = None,
) -> AsyncIterator[dict]:
    """Streaming RAG javob (tarix bilan)."""
    # Bo'sh scope — sof LLM, RAG yo'q
    if not scope or not _resolve_collections(scope):
        hist_text = _format_history(history)
        fb_prompt = f"{hist_text}\nYANGI SAVOL: {question}" if hist_text else question
        yield {"type": "mode", "mode": "llm_only"}
        yield {"type": "sources", "sources": []}
        async for tok in llm.stream(fb_prompt, system=_fallback_system(system_prompt)):
            yield {"type": "token", "text": tok}
        yield {"type": "done"}
        return

    collections = _resolve_collections(scope)
    q_vec = await embedder.embed(question)
    filter_: dict = {}
    # doc_ids list bo'lsa — MatchAny, doc_id bitta — MatchValue
    if doc_ids:
        filter_["doc_id"] = doc_ids
    elif doc_id:
        filter_["doc_id"] = doc_id
    if user_id is not None:
        filter_["user_id"] = user_id

    # Top-K ni dinamik: agar koʻp hujjat bor bo'lsa, qidiruvni kengaytirish
    base_top_k = runtime_config.get("top_k", settings.TOP_K)
    if doc_ids and len(doc_ids) > 1:
        # Har bir hujjatdan kamida 1 ta chunk olishga harakat
        effective_top_k = max(base_top_k, min(8, len(doc_ids) * 2))
    else:
        effective_top_k = base_top_k

    hits = await _gather_hits(
        q_vec, collections, top_k_each=effective_top_k,
        filter_=filter_ or None, final_top_k=effective_top_k,
    )

    # Similarity past bo'lsa (meta savol, kontekstda mos kelmaydi) — fallback
    LOW_SIMILARITY_THRESHOLD = 0.20
    if hits and (hits[0].get("score") or 0) < LOW_SIMILARITY_THRESHOLD:
        hits = []

    if not hits:
        if not allow_fallback:
            yield {"type": "mode", "mode": "no_context"}
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "text": "Berilgan hujjatlarda bu savolga javob topilmadi."}
            yield {"type": "done"}
            return
        # Fallback — tarix bilan
        hist_text = _format_history(history)
        fb_prompt = f"{hist_text}\nYANGI SAVOL: {question}" if hist_text else question
        yield {"type": "mode", "mode": "llm_only"}
        yield {"type": "sources", "sources": []}
        async for tok in llm.stream(fb_prompt, system=_fallback_system(system_prompt)):
            yield {"type": "token", "text": tok}
        yield {"type": "done"}
        return

    prompt, sources = _build_prompt(question, hits, history=history)
    sys_msg = RAG_SYSTEM
    if system_prompt:
        sys_msg = f"{RAG_SYSTEM}\n\nQo'shimcha:\n{system_prompt}"

    yield {"type": "mode", "mode": "rag"}
    yield {"type": "sources", "sources": sources}
    async for tok in llm.stream(prompt, system=sys_msg):
        yield {"type": "token", "text": tok}
    yield {"type": "done"}
