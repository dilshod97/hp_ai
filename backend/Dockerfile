FROM python:3.11-slim

WORKDIR /app

# Tizim paketlari (PDF, OCR, docx uchun)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-uzb \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    tesseract-ocr-uzb-cyrl \
    fontconfig \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
