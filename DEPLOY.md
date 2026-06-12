# 🚀 HP-AI audit assistant — Server'ga deploy qoʻllanmasi

## 📋 Server talablari

- **OS:** Ubuntu 22.04+ yoki Debian 12+ (yoki istalgan Linux)
- **GPU:** 24+ GB VRAM (NVIDIA — RTX 4090, A100, H100 va h.k.)
- **RAM:** 16+ GB
- **Disk:** 100+ GB (modellar, hujjatlar uchun)
- **Tarmoq:** Ollama porti uchun (11434/11435)

## 🛠️ 1-bosqich: Server tayyorlash

### Docker va Docker Compose

```bash
# Docker oʻrnatish
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Docker Compose (V2 plugin Docker bilan birga keladi, lekin tekshiring)
docker compose version
# Yoki eski v1:
sudo apt install docker-compose-plugin
```

### Ollama oʻrnatish (GPU bilan)

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Ollama sozlamalari (KUCHLI TAVSIYA!)
sudo systemctl edit ollama
```

Quyidagi qatorlarni qoʻshing:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11435"
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
```

Saqlab, qayta ishga tushiring:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Modellarni yuklab olish

```bash
# Asosiy LLM (12.85 GB)
ollama pull gpt-oss:20b

# Embedding modeli (1.21 GB)
ollama pull bge-m3
```

## 📦 2-bosqich: Loyihani GitHub'dan klonlash

```bash
cd /opt   # yoki /home/$USER
git clone https://github.com/dilshod97/hp_ai.git
cd hp_ai
```

## ⚙️ 3-bosqich: Sozlamalar

### `.env` faylni tayyorlash

```bash
cp .env.example .env
nano .env
```

Quyidagi qatorlarni sozlang:

```env
# Ollama URL — server'ning oʻzida bo'lsa:
OLLAMA_BASE_URL=http://host.docker.internal:11435

# Yoki alohida IP da boʻlsa:
# OLLAMA_BASE_URL=http://172.16.30.225:11435

# Modellar
LLM_MODEL=gpt-oss:20b
EMBEDDING_MODEL=bge-m3

# Auth (PRODUCTION'DA OʻZGARTIRING!)
JWT_SECRET=<32+ belgili tasodifiy satr>
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<kuchli parol>

# Boshqalari standartda qoldiriladi
```

**Tasodifiy JWT_SECRET yaratish:**
```bash
openssl rand -hex 32
```

### Data papkalarni tayyorlash

```bash
mkdir -p data/{uploads,laws,reports,datasets,trained_models}
mkdir -p qdrant_storage
```

## 🚀 4-bosqich: Ishga tushirish

```bash
# Birinchi marta — build qilamiz
docker compose up -d --build

# Loglarni kuzatish
docker compose logs -f backend
```

Tayyor xabarini kuting:
```
🚀 HP-AI audit assistant ishga tushyapti...
✓ SQLite tayyor: /data/hp_ai.db
✓ Ollama tayyor: http://host.docker.internal:11435
✓ llm mavjud: gpt-oss:20b
✓ embedding mavjud: bge-m3
✓ Embedding dimension aniqlandi: bge-m3 = 1024
✓ Qdrant collection tayyor: laws
✓ Qdrant collection tayyor: audit_reports
✓ Qdrant collection tayyor: uploads
✅ Backend tayyor — http://localhost:8000/docs
```

## 🌐 5-bosqich: Web kirishni sozlash (Nginx)

Tashqi tarmoqdan kirish uchun nginx reverse proxy:

```bash
sudo apt install nginx
sudo nano /etc/nginx/sites-available/hp-ai
```

```nginx
server {
    listen 80;
    server_name hp-ai.example.com;   # yoki IP manzilingiz

    # Maksimal yuklash hajmi (katta qonun fayllar uchun)
    client_max_body_size 500M;

    # Frontend
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API (CORS allaqachon yoqilgan)
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;          # SSE streaming uchun
        proxy_read_timeout 7200s;     # 2 soat (uzun ingest uchun)
    }

    location /docs {
        proxy_pass http://localhost:8000;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/hp-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### HTTPS (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d hp-ai.example.com
```

## ✅ 6-bosqich: Sinov

Brauzerda oching: `http://hp-ai.example.com` (yoki server IP'si)

1. **Login:** `admin` / `<.env'dagi parol>`
2. **🏛️ Asosiy bilim bazasi** — qonunlarni yuklash
3. **Chat** — sinab koʻrish

## 📤 7-bosqich: Lokal Mac'dan oʻzgarishlarni yuborish (`git push`)

Sizning Mac'da:

```bash
cd /Users/dilshodbaxriddinov/hp_project/hp_ai

# Birinchi marta — git init
git init
git remote add origin https://github.com/dilshod97/hp_ai.git
git branch -M main

# Hamma fayllarni qoʻshish
git add .
git commit -m "Initial commit — HP-AI audit assistant v1.0"
git push -u origin main
```

Keyingi safar oʻzgarish kiritsangiz:
```bash
git add .
git commit -m "Oʻzgarish tavsifi"
git push
```

Server'da yangilash:
```bash
cd /opt/hp_ai
git pull
docker compose up -d --build backend
```

## 🔒 Xavfsizlik tavsiyalari

1. **JWT_SECRET** — kamida 32 belgili tasodifiy satr
2. **ADMIN_PASSWORD** — kuchli, 16+ belgili
3. **HTTPS** — Let's Encrypt bilan
4. **Firewall:**
   ```bash
   sudo ufw allow 80,443/tcp
   sudo ufw allow 22/tcp
   sudo ufw enable
   ```
5. **Ollama portni** ishlab chiqarish muhitida **lokal hostda saqlang** (0.0.0.0 emas, 127.0.0.1)
6. **Maxfiy fayllarni** Git'ga **qoʻshmang** (`.env`, `data/`, `qdrant_storage/`)

## 📦 Backup

```bash
# Kunlik backup skripti
nano /opt/hp_ai/backup.sh
```

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M)
BACKUP_DIR=/var/backups/hp_ai
mkdir -p $BACKUP_DIR

# SQLite + config
tar czf $BACKUP_DIR/data_$DATE.tar.gz -C /opt/hp_ai data/hp_ai.db data/workspaces.json data/config.json

# Qdrant vectors
docker exec hp_qdrant tar czf - /qdrant/storage > $BACKUP_DIR/qdrant_$DATE.tar.gz

# 30 kundan eski backuplarni o'chirish
find $BACKUP_DIR -mtime +30 -delete

echo "✓ Backup tayyor: $DATE"
```

```bash
chmod +x /opt/hp_ai/backup.sh

# Crontab — har kuni 02:00 da
crontab -e
# Quyidagini qoʻshing:
0 2 * * * /opt/hp_ai/backup.sh >> /var/log/hp_ai_backup.log 2>&1
```

## 🐛 Muammolarni hal qilish

### Backend ishga tushmaydi

```bash
docker compose logs backend --tail=100
```

### Ollama'ga ulanmadi

```bash
# Server'da Ollama ishlayaptimi:
curl http://localhost:11435/api/tags

# Modellar:
ollama list

# Restart:
sudo systemctl restart ollama
```

### Qdrant xato

```bash
# Qdrant statusi
docker compose ps qdrant
docker compose logs qdrant
```

### Toʻliq qayta boshlash

```bash
docker compose down
docker compose up -d --build
```

## 📞 Yordam

- API dokumentatsiya: `http://server/docs`
- README: `README.md`
- Issues: https://github.com/dilshod97/hp_ai/issues
