# 🏛️ HP-AI audit assistant

Hisob palatasi auditorlari uchun sun'iy intellekt yordamchisi. Toʻliq self-hosted, hammasi bepul.

**Kontseptsiya:**
- 📚 **Asosiy bilim bazasi** — admin tomonidan yuklangan qonunlar va eski audit hisobotlari (barcha auditor ko'radi)
- 💬 **Shaxsiy chatlar** — auditor o'z hujjatlarini yuklab tahlil qiladi (faqat oʻzi koʻradi)
- 🔗 **API integratsiya** — har bir chat uchun avtomatik endpoint, tashqi tizimlarga ulanish

**Texnologiyalar:** Ollama (gpt-oss:20b) + BGE-M3 + Qdrant + Redis + FastAPI + vanilla HTML.

## Imkoniyatlar

- ❓ Qonunlar va eski audit hisobotlari asosida savol-javob (citation bilan)
- 📄 PDF / DOCX / XLSX / TXT yuklab, ichidan ma'lumot topish
- 🔍 Ikkita hujjatni solishtirish (farqlarni aniqlash)
- ⚡ Semantik kesh (takroriy savollar uchun darhol javob)
- 🌊 Streaming javob (token-by-token)
- 🎯 Qidiruv doirasi: qonunlar / hisobotlar / yuklangan fayllar

## Talab

Sizning server:
- 48 GB GPU
- Ollama oʻrnatilgan, ishga tushgan (`localhost:11434`)
- `gpt-oss:20b` modeli yuklangan
- Docker + Docker Compose

## Tezkor ishga tushirish

### 1) Ollama'ni ishga tushiring

Mac'da:
```bash
ollama serve   # alohida terminalda, agar app sifatida ochilmagan bo'lsa
```

> ⚙️ Kerakli modellar (`gpt-oss:20b`, `bge-m3`) **avtomatik yuklab olinadi**
> backend birinchi marta startup paytida. Internet tezligiga qarab
> 5–20 daqiqa olishi mumkin. Progress backend loglarida koʻrinadi.

### 2) Loyihani ishga tushiring

```bash
cd hp_ai
cp .env.example .env   # agar hali nusxalanmagan bo'lsa
docker-compose up -d --build
```

Servislar:
- **Frontend:** http://localhost:8080
- **Backend API:** http://localhost:8000 (Swagger: `/docs`)
- **Qdrant UI:** http://localhost:6333/dashboard

### 3) Modellar yuklanishini kuzating

```bash
docker-compose logs -f backend
```

Quyidagi loglarni koʻrasiz:
```
⏳ Ollama kutilyapti...
✅ Ollama tayyor
📥 Model yuklanyapti: bge-m3 (bu 5-15 daqiqa olishi mumkin)
   ⏳ bge-m3: 30% (0.20 / 0.66 GB)
✅ bge-m3 yuklab olindi
✓ llm mavjud: gpt-oss:20b
🔥 Warm-up: gpt-oss:20b
✅ Backend tayyor — http://localhost:8000/docs
```

### 4) Hujjatlarni indekslang

**Variant A — UI orqali:** brauzerda http://localhost:8080 ga kiring, faylni sudrab tashlang.

**Variant B — papkadan ommaviy:**

```bash
# Qonunlarni qoʻying:
cp /siz/qonunlar/*.pdf data/laws/
# Hisobotlarni qoʻying:
cp /siz/hisobotlar/*.pdf data/reports/

# Indekslash:
make ingest-laws
make ingest-reports
```

### 4) Savol bering

UI dan yoki API orqali:

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Shartnoma summasi qancha?", "scope": ["uploads"]}'
```

## API endpointlari

| Endpoint | Maqsad |
|---|---|
| `GET  /api/health` | Servislar holatini tekshirish |
| `POST /api/ask` | Sinxron savol-javob (JSON) |
| `POST /api/ask/stream` | Streaming javob (SSE) |
| `POST /api/documents/upload` | Fayl yuklash + indekslash |
| `POST /api/documents/ingest_dir` | Server papkasidan ingest |
| `POST /api/compare` | Ikki hujjatni solishtirish |
| `GET  /api/models` | Mavjud Ollama modellari ro'yxati |
| `POST /api/models/ensure` | Sozlamadagi modellarni avtomatik yuklab olish |
| `POST /api/models/pull` | Aniq modelni yuklab olish (streaming progress) |
| `GET  /api/config` | Joriy sozlamalar + mavjud modellar |
| `POST /api/config` | Sozlamalarni yangilash (model tanlash, temperature, top-k) |
| `POST /api/config/reset` | Standart sozlamalarga qaytarish |

Toʻliq dokumentatsiya: http://localhost:8000/docs

## Loyiha tuzilishi

```
hp_ai/
├── docker-compose.yml      # Qdrant + Redis + Backend + Frontend
├── .env                    # Sozlamalar
├── Makefile                # Yordamchi buyruqlar
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py         # FastAPI kirish nuqtasi
│       ├── config.py       # Sozlamalar
│       ├── api/            # HTTP endpointlar
│       │   ├── chat.py
│       │   ├── documents.py
│       │   ├── compare.py
│       │   └── health.py
│       ├── core/           # Biznes logika
│       │   ├── embeddings.py
│       │   ├── vector_store.py
│       │   ├── llm.py
│       │   ├── cache.py
│       │   ├── parser.py
│       │   ├── ingest.py
│       │   └── rag.py
│       └── models/
│           └── schemas.py
├── frontend/
│   └── index.html          # Chat UI
├── scripts/
│   └── ingest.py           # CLI ingest
└── data/
    ├── laws/               # Qonunlar (PDF/DOCX)
    ├── reports/            # Eski audit hisobotlari
    └── uploads/            # Foydalanuvchi yuklagan fayllar
```

## Arxitektura

```
┌──────────┐    ┌─────────┐    ┌────────┐
│ Frontend │───▶│ Backend │───▶│ Ollama │  (LLM + Embedding)
│  (HTML)  │    │(FastAPI)│    └────────┘
└──────────┘    │         │    ┌────────┐
                │         │───▶│ Qdrant │  (Vector DB)
                │         │    └────────┘
                │         │    ┌────────┐
                │         │───▶│ Redis  │  (Semantic cache)
                └─────────┘    └────────┘
```

## Sozlamalar (`.env`)

| Oʻzgaruvchi | Standart | Tushuntirish |
|---|---|---|
| `LLM_MODEL` | `gpt-oss:20b` | Ollama modeli |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding modeli |
| `CHUNK_SIZE` | `600` | Chunk hajmi (soʻzlarda) |
| `TOP_K` | `4` | Necha manba olinadi |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Kesh uchun oʻxshashlik chegarasi |
| `LLM_TEMPERATURE` | `0.1` | Past = aniq (huquqiy savollar uchun) |

## Tezlik

48 GB GPU da kutilgan koʻrsatkichlar:

| Hodisa | Vaqt |
|---|---|
| Birinchi token | 300-500 ms |
| To'liq javob (200 token) | 3-5 sek |
| Kesh hit | ~50 ms |

## Yordamchi buyruqlar

```bash
make up              # Hammasini ishga tushirish
make down            # Toʻxtatish
make logs            # Backend loglarini koʻrish
make ps              # Konteynerlar holati
make health          # /api/health chaqirish
make ingest-laws     # data/laws/ dan ingest
make ingest-reports  # data/reports/ dan ingest
make rebuild         # Backend ni qayta qurish
```

## Foydalanish bo'yicha maslahatlar

1. **Citation [N] formatini majburiy qildik** — modeldan har bir tasdiqdan keyin manba raqamini ko'rsatishi so'raladi.
2. **Temperature 0.1** — model ixtiro qilmaydi, faqat hujjatdagi maʼlumotni qaytaradi.
3. **Filter** — agar foydalanuvchi yuklagan fayldan savol bersa, scope = `["uploads"]` va `doc_id` yuboriladi.
4. **Kesh** — bir xil savol takror berilsa, darhol javob (LLM chaqirilmaydi).
5. **Streaming** — javob token-by-token kelganda, foydalanuvchi kutmasdan o'qiy boshlaydi.

## Keyingi qadamlar (tavsiya)

- [ ] Reranker qoʻshish (`bge-reranker-v2-m3`) — sifatni oshirish
- [ ] PostgreSQL — audit history va foydalanuvchilar
- [ ] Auth (Keycloak / oddiy JWT)
- [ ] LangFuse — LLM kuzatish
- [ ] Fine-tuning (QLoRA + Unsloth) — agar 1000+ savol-javob jufti bo'lsa
- [ ] OCR (skanerdan oʻtgan PDFlar uchun)

## Litsenziya

Ichki foydalanish uchun.
