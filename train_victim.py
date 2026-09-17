
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)


# ─── Argument Parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a victim causal-LM on raw or sanitized PII corpus."
    )
    p.add_argument("--model",          default="Qwen/Qwen2.5-0.5B",
                   help="Base causal-LM (NOT instruct variant)")
    p.add_argument("--corpus",         required=True,
                   help="JSONL with {doc_id, text, n_reps, is_canary, ...}")
    p.add_argument("--heldout",        required=True,
                   help="JSONL of never-trained docs for perplexity eval")
    p.add_argument("--target_modules", required=True,
                   help="Comma-separated LoRA target modules")
    p.add_argument("--lora_r",         type=int,   default=32)
    p.add_argument("--lora_alpha",     type=int,   default=64)
    p.add_argument("--lora_dropout",   type=float, default=0.05)
    p.add_argument("--epochs",         type=int,   default=4,
                   help="More epochs → higher memorization (for attack research)")
    p.add_argument("--per_device_batch", type=int, default=4)
    p.add_argument("--grad_accum",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--max_length",     type=int,   default=512)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--report_json",    default=None,
                   help="Optional JSON file to write training summary")
    return p.parse_args()


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_corpus_dataset(records, tokenizer, max_length):

    texts = [r["text"] for r in records]

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


# ─── Perplexity Computation ──────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, records, max_length=512, device=None):

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss  = 0.0
    total_tokens = 0

    with torch.no_grad():
        for rec in records:
            text = rec["text"]
            enc  = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=max_length)
            input_ids = enc.input_ids.to(device)
            n_tokens  = input_ids.shape[1]

            if n_tokens < 2:
                continue

            out = model(input_ids, labels=input_ids)
            # out.loss is mean cross-entropy over non-masked tokens
            # Multiply back to get total token loss
            total_loss   += out.loss.item() * (n_tokens - 1)
            total_tokens += n_tokens - 1     # teacher-forcing: predict tokens 1…N-1

    if total_tokens == 0:
        return float("inf")

    mean_nll = total_loss / total_tokens
    ppl      = math.exp(mean_nll)
    return ppl


# ─── Model + PEFT ────────────────────────────────────────────────────────────

def load_base_model(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer


def apply_lora(model, args):

    target_modules = [m.strip() for m in args.target_modules.split(",")]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        inference_mode=False,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ─── Corpus Sanitizer Helper ─────────────────────────────────────────────────

def sanitize_corpus(raw_corpus_path: str, sanitizer_model, sanitizer_tokenizer,
                    out_path: str, max_new_tokens: int = 512):

    from sanitize_eval import SYSTEM_PROMPT, LABEL_RE

    records = []
    with open(raw_corpus_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Sanitizing {len(records)} corpus rows …")

    sanitizer_model.eval()

    with open(out, "w", encoding="utf-8") as fh_out:
        for i, rec in enumerate(records):
            if i % 500 == 0:
                print(f"    [{i}/{len(records)}]", flush=True)

            src = rec["text"]

            # Build inference prompt using the sanitizer's chat format
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": src},
            ]
            prompt = sanitizer_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = sanitizer_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(sanitizer_model.device)

            with torch.no_grad():
                out_ids = sanitizer_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=sanitizer_tokenizer.eos_token_id,
                )

            new_tokens = out_ids[0][inputs.input_ids.shape[1]:]
            sanitized_text = sanitizer_tokenizer.decode(new_tokens, skip_special_tokens=True)

            new_rec = dict(rec)
            new_rec["text"] = sanitized_text
            fh_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    print(f"  Saved sanitized corpus to {out}")


# ─── Training ────────────────────────────────────────────────────────────────

def train_victim(model, tokenizer, train_ds, args):

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,       # Causal LM (autoregressive), NOT masked LM
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="no",
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    t0 = time.perf_counter()
    result = trainer.train()
    wall_s = time.perf_counter() - t0

    trainer.save_model()
    return result.training_loss, wall_s


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    victim_type = "V-san" if "san" in args.output_dir else "V-raw"
    print(f"\n{'='*60}")
    print(f"  Training victim model: {victim_type}")
    print(f"  Corpus: {args.corpus}")
    print(f"  Epochs: {args.epochs}  |  r={args.lora_r}  α={args.lora_alpha}")
    print(f"{'='*60}\n")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("[1/5] Loading corpus …")
    corpus_records  = load_jsonl(args.corpus)
    heldout_records = load_jsonl(args.heldout)
    print(f"      Corpus: {len(corpus_records)} physical rows")
    print(f"      Heldout: {len(heldout_records)} docs")

    # ── Step 2: Load model ───────────────────────────────────────────────────
    print("[2/5] Loading base model …")
    model, tokenizer = load_base_model(args)

    # ── Step 3: Baseline perplexity (pre-training) ───────────────────────────
    print("[3/5] Computing baseline (pre-training) perplexity on heldout …")
    base_ppl = compute_perplexity(model, tokenizer, heldout_records[:100],
                                  max_length=args.max_length)
    print(f"      Baseline PPL = {base_ppl:.2f}")

    # ── Step 4: Apply LoRA + Train ───────────────────────────────────────────
    print("[4/5] Applying LoRA and training …")
    model = apply_lora(model, args)

    train_ds = build_corpus_dataset(corpus_records, tokenizer, args.max_length)
    final_train_loss, wall_s = train_victim(model, tokenizer, train_ds, args)

    # ── Step 5: Held-out perplexity (post-training) ──────────────────────────
    print("[5/5] Computing post-training perplexity on heldout …")
    heldout_ppl = compute_perplexity(model, tokenizer, heldout_records[:100],
                                     max_length=args.max_length)
    print(f"      Post-training PPL = {heldout_ppl:.2f}")
    print(f"      PPL change: {base_ppl:.2f} → {heldout_ppl:.2f} "
          f"({'↑ worse' if heldout_ppl > base_ppl else '↓ better'})")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "victim_type":      victim_type,
        "corpus":           args.corpus,
        "corpus_rows":      len(corpus_records),
        "epochs":           args.epochs,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
        "target_modules":   args.target_modules,
        "final_train_loss": round(final_train_loss, 6),
        "wall_s":           round(wall_s, 1),
        "baseline_ppl":     round(base_ppl, 3),
        "heldout_ppl":      round(heldout_ppl, 3),
    }

    print(f"\n{'─'*50}")
    for k, v in summary.items():
        print(f"  {k:25s} = {v}")
    print(f"{'─'*50}\n")

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  Report → {out}")


if __name__ == "__main__":
    main()
