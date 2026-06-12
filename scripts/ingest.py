"""CLI: papkadagi qonun/hisobotlarni ommaviy ingest qilish.

Foydalanish:
    python scripts/ingest.py --target laws --path /data/laws
    python scripts/ingest.py --target reports --path /data/reports
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Backend kodi PYTHONPATH ga qoʻshilsin
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.core.ingest import ingest_directory  # noqa: E402


COLLECTIONS = {
    "laws": settings.QDRANT_COLLECTION_LAWS,
    "reports": settings.QDRANT_COLLECTION_REPORTS,
    "uploads": settings.QDRANT_COLLECTION_UPLOADS,
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(COLLECTIONS.keys()))
    ap.add_argument("--path", required=True)
    args = ap.parse_args()

    col = COLLECTIONS[args.target]
    print(f"→ Ingest: {args.path} -> collection '{col}'")
    res = await ingest_directory(args.path, col)
    ok = sum(1 for r in res if r.get("chunks", 0) > 0)
    print(f"✅ {ok}/{len(res)} fayl indekslandi")
    for r in res:
        if "error" in r:
            print(f"  ❌ {r.get('filename')}: {r['error']}")
        else:
            print(f"  ✓ {r.get('filename')} — {r.get('chunks')} chunk")


if __name__ == "__main__":
    asyncio.run(main())
