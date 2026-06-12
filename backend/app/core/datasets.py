"""Training dataset CRUD + JSONL eksport + chat history'dan generate."""
import json
import os
from datetime import datetime

from sqlalchemy import delete, select

from app.config import settings
from app.core.db import Dataset, DatasetItem, Message, SessionLocal

DATASETS_DIR = os.path.join(settings.DATA_DIR, "datasets")


def _ensure_dir() -> None:
    os.makedirs(DATASETS_DIR, exist_ok=True)


# ---------- Dataset CRUD ----------

async def create_dataset(
    name: str, description: str | None = None, owner_id: int | None = None
) -> dict:
    async with SessionLocal() as s:
        ds = Dataset(name=name, description=description, owner_id=owner_id)
        s.add(ds)
        await s.commit()
        await s.refresh(ds)
        return ds.to_dict()


async def list_datasets(owner_id: int | None = None) -> list[dict]:
    async with SessionLocal() as s:
        q = select(Dataset).order_by(Dataset.id.desc())
        if owner_id is not None:
            q = q.where(Dataset.owner_id == owner_id)
        rows = (await s.execute(q)).scalars().all()
        result = []
        for r in rows:
            d = r.to_dict()
            # Item count
            cnt = await s.execute(
                select(DatasetItem).where(DatasetItem.dataset_id == r.id)
            )
            d["item_count"] = len(cnt.scalars().all())
            result.append(d)
        return result


async def get_dataset(dataset_id: int) -> dict | None:
    async with SessionLocal() as s:
        ds = await s.get(Dataset, dataset_id)
        return ds.to_dict() if ds else None


async def delete_dataset(dataset_id: int) -> bool:
    async with SessionLocal() as s:
        ds = await s.get(Dataset, dataset_id)
        if not ds:
            return False
        # Itemslarni ham o'chiramiz
        await s.execute(delete(DatasetItem).where(DatasetItem.dataset_id == dataset_id))
        await s.delete(ds)
        await s.commit()
        return True


# ---------- DatasetItem CRUD ----------

async def add_item(
    dataset_id: int,
    instruction: str,
    output: str,
    input_text: str | None = None,
    source_message_id: int | None = None,
    quality: int = 3,
) -> dict:
    async with SessionLocal() as s:
        item = DatasetItem(
            dataset_id=dataset_id,
            instruction=instruction,
            input_text=input_text,
            output=output,
            source_message_id=source_message_id,
            quality=quality,
        )
        s.add(item)
        await s.commit()
        await s.refresh(item)
        return item.to_dict()


async def list_items(dataset_id: int, limit: int = 1000) -> list[dict]:
    async with SessionLocal() as s:
        q = (
            select(DatasetItem)
            .where(DatasetItem.dataset_id == dataset_id)
            .order_by(DatasetItem.id.asc())
            .limit(limit)
        )
        rows = (await s.execute(q)).scalars().all()
        return [r.to_dict() for r in rows]


async def update_item(item_id: int, **fields) -> dict | None:
    async with SessionLocal() as s:
        item = await s.get(DatasetItem, item_id)
        if not item:
            return None
        for k, v in fields.items():
            if k == "input":
                item.input_text = v
            elif hasattr(item, k):
                setattr(item, k, v)
        await s.commit()
        await s.refresh(item)
        return item.to_dict()


async def delete_item(item_id: int) -> bool:
    async with SessionLocal() as s:
        item = await s.get(DatasetItem, item_id)
        if not item:
            return False
        await s.delete(item)
        await s.commit()
        return True


# ---------- Chat history -> Dataset ----------

async def import_from_chat(
    dataset_id: int,
    workspace_id: str | None = None,
    only_good: bool = True,
    only_liked: bool = False,
) -> int:
    """Chat tarixidagi user-bot juftliklarini dataset'ga qoʻshish.

    only_liked=True — faqat 👍 bosilgan javoblar (eng sifatli, RLHF-style).
    only_good=True — mode='rag', bo'sh emas, xatosiz javoblar.
    """
    async with SessionLocal() as s:
        q = select(Message).order_by(Message.id.asc())
        if workspace_id:
            q = q.where(Message.workspace_id == workspace_id)
        msgs = (await s.execute(q)).scalars().all()

        # Takroriy import oldini olish — allaqachon datasetdagi message_id'lar
        existing_q = select(DatasetItem.source_message_id).where(
            DatasetItem.dataset_id == dataset_id,
            DatasetItem.source_message_id.isnot(None),
        )
        existing_ids = {r[0] for r in (await s.execute(existing_q)).all()}

        added = 0
        prev_user = None
        for m in msgs:
            if m.role == "user":
                prev_user = m
            elif m.role == "bot" and prev_user is not None:
                if m.id in existing_ids:
                    prev_user = None
                    continue
                # Like filtri — eng yuqori ustunlik
                if only_liked:
                    if m.rating != 1:
                        prev_user = None
                        continue
                elif only_good:
                    # Dislike bosilganlarni hech qachon olmaymiz
                    if m.rating == -1:
                        prev_user = None
                        continue
                    if m.mode != "rag":
                        prev_user = None
                        continue
                    if not (m.text or "").strip():
                        prev_user = None
                        continue
                    if "xato" in (m.text or "").lower()[:50]:
                        prev_user = None
                        continue
                if not (m.text or "").strip():
                    prev_user = None
                    continue
                # Item yaratish — like bosilganlar eng yuqori sifat
                item = DatasetItem(
                    dataset_id=dataset_id,
                    instruction=prev_user.text,
                    input_text=None,
                    output=m.text,
                    source_message_id=m.id,
                    quality=5 if m.rating == 1 else (4 if m.mode == "rag" else 3),
                )
                s.add(item)
                added += 1
                prev_user = None
        await s.commit()
        return added


# ---------- JSONL eksport ----------

async def export_jsonl(dataset_id: int) -> str:
    """Dataset'ni JSONL faylga eksport qiladi (training script uchun)."""
    _ensure_dir()
    items = await list_items(dataset_id, limit=100000)
    ds = await get_dataset(dataset_id)
    name = (ds or {}).get("name", f"dataset_{dataset_id}").replace("/", "_")
    fname = f"{name}_{dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    path = os.path.join(DATASETS_DIR, fname)

    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            obj = {
                "instruction": it["instruction"],
                "input": it.get("input") or "",
                "output": it["output"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return path


# ---------- JSONL import ----------

async def import_jsonl(dataset_id: int, path: str) -> int:
    """JSONL fayldan dataset'ga import qilish."""
    added = 0
    async with SessionLocal() as s:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                instr = obj.get("instruction") or obj.get("prompt") or ""
                out = obj.get("output") or obj.get("response") or obj.get("answer") or ""
                inp = obj.get("input")
                if not instr or not out:
                    continue
                item = DatasetItem(
                    dataset_id=dataset_id,
                    instruction=instr,
                    input_text=inp,
                    output=out,
                    quality=3,
                )
                s.add(item)
                added += 1
        await s.commit()
    return added
