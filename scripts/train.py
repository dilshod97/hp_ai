"""QLoRA fine-tuning skripti (Unsloth + TRL).

Bu skript GPU serverda ishlatilishi kerak (Mac'da ishlamaydi).
Backend container'dan chaqirilganda — agar GPU mavjud bo'lsa, ishlaydi.
Aks holda — fail (xato xabar bilan).

Foydalanish:
    python train.py --dataset /data/datasets/foo.jsonl \\
                    --base-model gpt-oss:20b \\
                    --output /data/trained_models/my_model \\
                    --epochs 3 --batch-size 2 --lora-r 16

Talab:
    pip install unsloth trl peft transformers torch bitsandbytes accelerate

Progress chiqishi (PROGRESS 0.XX) — backend tracker uchun.
"""
import argparse
import json
import os
import sys
import time


def log(msg: str) -> None:
    print(msg, flush=True)


def progress(pct: float) -> None:
    """Backend tracker uchun maxsus format."""
    print(f"PROGRESS {pct:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="JSONL fayl")
    ap.add_argument("--base-model", default="gpt-oss:20b")
    ap.add_argument("--output", required=True, help="Output papka (LoRA adapter)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    log("=" * 60)
    log("🎓 QLoRA Fine-tuning")
    log("=" * 60)
    log(f"Dataset:    {args.dataset}")
    log(f"Base:       {args.base_model}")
    log(f"Output:     {args.output}")
    log(f"Epochs:     {args.epochs}")
    log(f"Batch:      {args.batch_size}")
    log(f"LoRA r:     {args.lora_r}")
    log(f"LR:         {args.lr}")
    log("")

    # 1) GPU tekshirish
    progress(0.02)
    try:
        import torch
    except ImportError:
        log("❌ PyTorch o'rnatilmagan. Server'da o'rnating:")
        log("   pip install torch transformers peft trl unsloth bitsandbytes accelerate")
        sys.exit(1)

    if not torch.cuda.is_available():
        log("❌ GPU topilmadi. Training faqat NVIDIA GPU bilan ishlaydi.")
        log("   nvidia-smi bilan tekshiring.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log(f"✓ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    log("")

    # 2) Dataset yuklash
    progress(0.05)
    log("📚 Dataset yuklanmoqda...")
    if not os.path.exists(args.dataset):
        log(f"❌ Dataset topilmadi: {args.dataset}")
        sys.exit(1)

    examples = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except Exception:
                continue
    log(f"✓ {len(examples)} ta misol yuklandi")

    if len(examples) < 10:
        log("⚠️ Juda kam misol (< 10). Sifatli fine-tune uchun 100+ tavsiya etiladi.")

    # 3) Unsloth + model
    progress(0.10)
    log("")
    log("🤖 Model yuklanmoqda (Unsloth, 4-bit)...")
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        log("❌ Unsloth o'rnatilmagan. Server'da:")
        log("   pip install unsloth")
        sys.exit(1)

    # Unsloth uchun model nomi — HuggingFace formatda
    # gpt-oss:20b -> openai/gpt-oss-20b
    base = args.base_model.replace(":", "-")
    if base.startswith("gpt-oss-"):
        hf_name = f"unsloth/{base}-unsloth-bnb-4bit"
    else:
        hf_name = base

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=hf_name,
            max_seq_length=args.max_seq_len,
            dtype=None,  # auto
            load_in_4bit=True,
        )
    except Exception as e:
        log(f"❌ Model yuklanmadi: {e}")
        log(f"   HuggingFace'da '{hf_name}' bormi tekshiring.")
        sys.exit(1)

    log("✓ Model yuklandi")

    # 4) LoRA adapter
    progress(0.20)
    log("")
    log("🔧 LoRA adapter qoʻshilmoqda...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    log("✓ LoRA tayyor")

    # 5) Dataset formatlash
    progress(0.25)
    log("")
    log("📝 Dataset formatlanyapti...")

    def format_example(ex: dict) -> str:
        instr = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")
        if inp:
            return (
                f"### Instruction:\n{instr}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{out}{tokenizer.eos_token}"
            )
        return (
            f"### Instruction:\n{instr}\n\n"
            f"### Response:\n{out}{tokenizer.eos_token}"
        )

    from datasets import Dataset

    texts = [format_example(ex) for ex in examples]
    train_ds = Dataset.from_dict({"text": texts})
    log(f"✓ {len(train_ds)} ta misol tayyor")

    # 6) Training
    progress(0.30)
    log("")
    log("🚂 Training boshlanyapti...")
    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Progressni 30% dan 95% ga olib boramiz
    start_progress = 0.30
    end_progress = 0.95
    total_steps = max(1, len(train_ds) // args.batch_size * args.epochs)

    class ProgressCallback:
        def __init__(self):
            self.last_pct = -1
        def on_step_end(self, args_, state, control, **kwargs):
            pct = start_progress + (state.global_step / total_steps) * (end_progress - start_progress)
            if int(pct * 100) != self.last_pct:
                progress(min(pct, end_progress))
                self.last_pct = int(pct * 100)

    # transformers Callback API
    from transformers import TrainerCallback

    class TrainerProgressCallback(TrainerCallback):
        last_pct = -1
        def on_step_end(self, args_, state, control, **kwargs):
            pct = start_progress + (state.global_step / total_steps) * (end_progress - start_progress)
            if int(pct * 100) != TrainerProgressCallback.last_pct:
                progress(min(pct, end_progress))
                TrainerProgressCallback.last_pct = int(pct * 100)

    trainer.add_callback(TrainerProgressCallback())

    t0 = time.time()
    trainer.train()
    duration = time.time() - t0
    log(f"✓ Training tugadi: {duration/60:.1f} daqiqa")

    # 7) Saqlash
    progress(0.95)
    log("")
    log("💾 LoRA adapter saqlanyapti...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # GGUF eksport (Ollama uchun) — Unsloth bilan
    try:
        log("🔄 GGUF formatga eksport (Ollama uchun)...")
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method="q4_k_m")
        log(f"✓ GGUF saqlandi: {output_dir}/")
    except Exception as e:
        log(f"⚠️ GGUF eksport bajarilmadi: {e}")
        log("   Qo'lda: llama.cpp convert.py orqali aylantirib bo'ladi")

    progress(1.0)
    log("")
    log("=" * 60)
    log(f"✅ TAYYOR! Adapter: {output_dir}")
    log("=" * 60)
    log("")
    log("Ollama'ga qo'shish:")
    log(f"  cd {output_dir}")
    log("  # Modelfile yarating:")
    log("  echo 'FROM ./unsloth.Q4_K_M.gguf' > Modelfile")
    log("  ollama create my-finetuned -f Modelfile")
    log("  ollama run my-finetuned")


if __name__ == "__main__":
    main()
