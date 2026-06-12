"""Ollama startup initializer.

Backend ishga tushganda:
1. Ollama serveriga ulanishni tekshiradi (kutadi, agar tayyor bo'lmasa)
2. Kerakli modellar (LLM + Embedding) borligini tekshiradi
3. Yo'q bo'lsa — avtomatik pull qiladi (progress log bilan)
4. Birinchi savol tez bo'lishi uchun modellarni warm-up qiladi
"""
import asyncio
import json
import logging

import httpx

from app.config import settings

log = logging.getLogger("ollama_init")


async def _wait_for_ollama(max_wait: int = 120) -> bool:
    """Ollama serveriga ulanishni kutadi."""
    waited = 0
    async with httpx.AsyncClient(timeout=5.0) as c:
        while waited < max_wait:
            try:
                r = await c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                if r.status_code == 200:
                    log.info("✅ Ollama tayyor: %s", settings.OLLAMA_BASE_URL)
                    return True
            except Exception:
                pass
            log.info("⏳ Ollama kutilyapti... (%ss / %ss)", waited, max_wait)
            await asyncio.sleep(3)
            waited += 3
    log.error("❌ Ollama %ss ichida javob bermadi: %s", max_wait, settings.OLLAMA_BASE_URL)
    return False


async def _list_models() -> set[str]:
    """Ollama da mavjud modellar ro'yxatini olish."""
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
        r.raise_for_status()
        data = r.json()
        # Ham 'gpt-oss:20b', ham 'gpt-oss' formatini qabul qilamiz
        names: set[str] = set()
        for m in data.get("models", []):
            name = m.get("name", "")
            names.add(name)
            if ":" in name:
                names.add(name.split(":")[0])
        return names


async def _pull_model(model: str) -> bool:
    """Modelni yuklash. Streaming progress log qiladi."""
    log.info("📥 Model yuklanyapti: %s (bu 5-15 daqiqa olishi mumkin)", model)
    last_status = ""
    last_percent = -1

    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream(
            "POST",
            f"{settings.OLLAMA_BASE_URL}/api/pull",
            json={"name": model, "stream": True},
        ) as r:
            if r.status_code != 200:
                body = await r.aread()
                log.error("❌ Pull xatosi (%s): %s", r.status_code, body[:300])
                return False

            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in obj:
                    log.error("❌ %s pull xatosi: %s", model, obj["error"])
                    return False

                status = obj.get("status", "")
                total = obj.get("total")
                completed = obj.get("completed")

                # Progressni 10% chegaralarida log qilamiz (terminal'ni to'ldirmaslik uchun)
                if total and completed:
                    percent = int(completed * 100 / total)
                    if percent // 10 != last_percent // 10:
                        gb_total = total / 1024**3
                        gb_done = completed / 1024**3
                        log.info(
                            "   ⏳ %s: %d%% (%.2f / %.2f GB)",
                            model, percent, gb_done, gb_total,
                        )
                        last_percent = percent
                elif status and status != last_status:
                    log.info("   • %s: %s", model, status)
                    last_status = status

                if obj.get("status") == "success":
                    log.info("✅ %s yuklab olindi", model)
                    return True
    return True


async def _warm_up(model: str) -> None:
    """Modelni VRAM ga yuklab qoʻyish (birinchi savol tez bo'lsin)."""
    log.info("🔥 Warm-up: %s", model)
    try:
        async with httpx.AsyncClient(timeout=180.0) as c:
            await c.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "prompt": "hi", "stream": False,
                      "options": {"num_predict": 1}},
            )
        log.info("✅ Warm-up tugadi: %s", model)
    except Exception as e:
        log.warning("⚠️  Warm-up bajarilmadi (%s): %s", model, e)


async def _warm_up_embedding(model: str) -> None:
    log.info("🔥 Embedding warm-up: %s", model)
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            await c.post(
                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": model, "prompt": "salom"},
            )
        log.info("✅ Embedding warm-up tugadi: %s", model)
    except Exception as e:
        log.warning("⚠️  Embedding warm-up bajarilmadi: %s", e)


async def ensure_models() -> None:
    """Asosiy entrypoint — barcha kerakli modellarni tekshiradi/yuklaydi."""
    ok = await _wait_for_ollama()
    if not ok:
        log.warning("Ollama bilan aloqa yo'q — modellar tekshirilmadi.")
        return

    required = [
        ("llm", settings.LLM_MODEL),
        ("embedding", settings.EMBEDDING_MODEL),
    ]

    try:
        existing = await _list_models()
    except Exception as e:
        log.error("Modellar roʻyxatini olib bo'lmadi: %s", e)
        return

    log.info("Mavjud modellar: %s", sorted(existing) or "(yo'q)")

    for kind, name in required:
        bare = name.split(":")[0]
        if name in existing or bare in existing:
            log.info("✓ %s mavjud: %s", kind, name)
            continue

        ok = await _pull_model(name)
        if not ok:
            log.error("❌ %s yuklanmadi: %s — backend baribir ishga tushadi", kind, name)

    # Warm-up — fonda, blocking emas
    asyncio.create_task(_warm_up(settings.LLM_MODEL))
    asyncio.create_task(_warm_up_embedding(settings.EMBEDDING_MODEL))
