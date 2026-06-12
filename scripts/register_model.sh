#!/bin/bash
# Fine-tuned modelni Ollama'ga ro'yxatdan o'tkazish.
#
# Foydalanish:
#   ./register_model.sh /data/trained_models/my_audit_model my_audit
#
# Argumentlar:
#   $1 — GGUF fayl bor papka
#   $2 — Ollama'da koʻrinadigan model nomi
set -e

MODEL_DIR="${1:?Usage: $0 <model_dir> <ollama_name>}"
OLLAMA_NAME="${2:?Usage: $0 <model_dir> <ollama_name>}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "❌ Papka topilmadi: $MODEL_DIR"
  exit 1
fi

# GGUF faylni topish
GGUF=$(find "$MODEL_DIR" -name "*.gguf" | head -1)
if [ -z "$GGUF" ]; then
  echo "❌ GGUF fayl topilmadi. Avval train.py ishga tushiring va GGUF eksport qiling."
  exit 1
fi

echo "📦 GGUF: $GGUF"
echo "🏷️  Ollama nomi: $OLLAMA_NAME"

# Modelfile yaratish
MODELFILE="$MODEL_DIR/Modelfile"
cat > "$MODELFILE" <<EOF
FROM $GGUF

# Hisob palatasi auditorlari uchun fine-tune qilingan
TEMPLATE """### Instruction:
{{ .System }}{{ if .Prompt }}
{{ .Prompt }}{{ end }}

### Response:
{{ .Response }}"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "### Instruction:"
PARAMETER stop "### Response:"

SYSTEM "Siz Hisob palatasi auditorlari uchun yordamchi sun'iy intellektsiz. Qonunlar va audit hisobotlari asosida aniq va manba ko'rsatilgan javob bering."
EOF

echo "✓ Modelfile yaratildi"

# Ollama'ga register qilish
echo "🚀 Ollama'ga register qilinmoqda..."
ollama create "$OLLAMA_NAME" -f "$MODELFILE"

echo ""
echo "✅ Tayyor! Endi modelni ishlatish mumkin:"
echo ""
echo "  Test:    ollama run $OLLAMA_NAME"
echo ""
echo "  UI:      ⚙️ Sozlamalar → Chat / LLM modeli → '$OLLAMA_NAME'"
