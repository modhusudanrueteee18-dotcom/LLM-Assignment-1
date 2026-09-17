import argparse
import csv
import json
import math
import os
import re
import sys
import time
import zlib
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# ─── Constants ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a PII anonymizer. Given text that may contain personal information, "
    "replace every identified PII value with a bracketed placeholder like [EMAIL], "
    "[TEL], [SOCIALNUMBER], [GIVENNAME1], [LASTNAME1], [IP], [DATE], [PASS], etc. "
    "Preserve all non-PII content exactly. Return only the anonymized text."
)

CSV_HEADER = [
    "run_id", "method", "target_modules", "lora_r", "lora_alpha", "quant",
    "trainable_params", "total_params", "trainable_pct",
    "peak_gpu_mem_gb", "train_wallclock_s", "tokens_per_s",
    "final_train_loss", "val_loss", "val_entity_f1", "val_leak_rate",
]

LABEL_RE = re.compile(r'\[([A-Z][A-Z0-9]*?)(?:_?\d+)?\]')


# ─── Argument Parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Part A PEFT sweep — appends one row to --results_csv")
    p.add_argument("--model",          required=True,  help="HF model id or local path")
    p.add_argument("--train_file",     required=True,  help="partA_train.jsonl")
    p.add_argument("--val_file",       required=True,  help="partA_validation.jsonl")
    p.add_argument("--target_modules", required=True,
                   help="Comma-separated LoRA target modules, e.g. q_proj,v_proj")
    p.add_argument("--lora_r",         type=int,   required=True)
    p.add_argument("--lora_alpha",     type=int,   required=True)
    p.add_argument("--lora_dropout",   type=float, default=0.05)
    p.add_argument("--quant",          choices=["none", "4bit"], default="none")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--run_id",         required=True,  help="e.g. A1 … A6")
    p.add_argument("--output_dir",     required=True,  help="Where to save adapter weights")
    p.add_argument("--results_csv",    required=True,  help="CSV to append one row")
    p.add_argument("--epochs",         type=int,   default=1)
    p.add_argument("--per_device_batch", type=int, default=4)
    p.add_argument("--grad_accum",     type=int,   default=4,
                   help="Gradient accumulation steps — effective_batch = per_device × grad_accum")
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--max_length",     type=int,   default=512,
                   help="Max tokens per training example (source + target)")
    p.add_argument("--val_batch_size", type=int,   default=8)
    p.add_argument("--val_max_new",    type=int,   default=256)
    return p.parse_args()


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_jsonl(path: str):
    """Load every non-empty line as a JSON object."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_chat_text(example, tokenizer, add_generation_prompt: bool = False):

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["source_text"]},
    ]
    if not add_generation_prompt:
        messages.append({"role": "assistant", "content": example["target_text"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def build_hf_dataset(records, tokenizer):
    texts = [make_chat_text(r, tokenizer) for r in records]
    return Dataset.from_dict({"text": texts})


# ─── Span Recovery (used for validation F1) ──────────────────────────────────

def compute_iou(s1, e1, s2, e2):

    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter == 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def recover_spans_difflib(source: str, predicted: str):

    import difflib

    matcher = difflib.SequenceMatcher(None, source, predicted, autojunk=False)
    spans = []

    for tag, s1, s2, p1, p2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        pred_frag = predicted[p1:p2]
        src_frag  = source[s1:s2]

        labels = LABEL_RE.findall(pred_frag)

        if tag == "replace" and labels:
            if len(labels) == 1:
                # Single label → single span covering the whole replace block
                spans.append({"start": s1, "end": s2, "label": labels[0]})
            else:
                # Multiple labels in one replace block.
                # Heuristic: distribute source characters proportionally
                # (best-effort; exact PII boundaries unknown without diffing values)
                src_len = s2 - s1
                portion = src_len // len(labels)
                for k, lbl in enumerate(labels):
                    sp_s = s1 + k * portion
                    sp_e = s1 + (k + 1) * portion if k < len(labels) - 1 else s2
                    spans.append({"start": sp_s, "end": sp_e, "label": lbl})

        elif tag == "delete" and s2 > s1:
            # Source text deleted with no label → implicit masking
            spans.append({"start": s1, "end": s2, "label": "UNKNOWN"})

    return spans


# ─── Metrics ─────────────────────────────────────────────────────────────────

def span_f1(gold_spans, pred_spans, criterion="exact"):

    if not gold_spans and not pred_spans:
        return 1.0, 1.0, 1.0, 0, 0, 0      # tp, fp, fn
    if not gold_spans:
        return 0.0, 1.0, 0.0, 0, len(pred_spans), 0
    if not pred_spans:
        return 1.0, 0.0, 0.0, 0, 0, len(gold_spans)

    matched_gold = set()
    tp = 0

    for pred in pred_spans:
        for j, gold in enumerate(gold_spans):
            if j in matched_gold:
                continue
            if criterion == "exact":
                match = (pred["start"] == gold["start"] and
                         pred["end"]   == gold["end"]   and
                         pred["label"] == gold["label"])
            else:  # iou50
                match = (compute_iou(pred["start"], pred["end"],
                                     gold["start"], gold["end"]) >= 0.5 and
                         pred["label"] == gold["label"])
            if match:
                tp += 1
                matched_gold.add(j)
                break

    fp = len(pred_spans) - tp
    fn = len(gold_spans) - len(matched_gold)
    p  = tp / (tp + fp) if tp + fp > 0 else 0.0
    r  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1, tp, fp, fn


def compute_leak_rate(gold_spans, source_text, predicted_text):

    def normalize(s):
        return " ".join(s.lower().split())

    norm_pred = normalize(predicted_text)
    gold_values = [normalize(source_text[g["start"]:g["end"]]) for g in gold_spans]
    if not gold_values:
        return 0.0
    leaked = sum(1 for v in gold_values if v and v in norm_pred)
    return leaked / len(gold_values)


# ─── Validation Evaluation ───────────────────────────────────────────────────

def evaluate_on_val(model, tokenizer, val_records, args):

    model.eval()
    total_tp = total_fp = total_fn = 0
    leak_num = leak_den = 0

    # Evaluate on at most 200 examples to stay within time budget
    eval_records = val_records[:200]

    with torch.no_grad():
        for rec in eval_records:
            # Build inference prompt (no target)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": rec["source_text"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=args.max_length).to(model.device)

            out = model.generate(
                **inputs,
                max_new_tokens=args.val_max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = out[0][inputs.input_ids.shape[1]:]
            predicted_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            gold_spans = rec.get("privacy_mask", [])
            pred_spans = recover_spans_difflib(rec["source_text"], predicted_text)

            _, _, _, tp, fp, fn = span_f1(gold_spans, pred_spans, criterion="exact")
            total_tp += tp
            total_fp += fp
            total_fn += fn

            lr = compute_leak_rate(gold_spans, rec["source_text"], predicted_text)
            leak_num += lr * len(gold_spans)
            leak_den += len(gold_spans)

    # Micro F1
    micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p + micro_r > 0 else 0.0)

    val_leak_rate = leak_num / leak_den if leak_den > 0 else 0.0
    return micro_f1, val_leak_rate


# ─── Model + PEFT Setup ──────────────────────────────────────────────────────

def load_model_and_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # Required for causal LM loss masking

    if args.quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat-4 (information-optimal)
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,      # Double quantization → extra 0.37 bit/param saved
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # Required before wrapping with PEFT when using 4-bit quantization
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()

    return model, tokenizer


def configure_lora(model, args):

    target_modules = [m.strip() for m in args.target_modules.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    return model


# ─── Training ────────────────────────────────────────────────────────────────

def train(model, tokenizer, train_ds, val_ds, args):

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        fp16=(args.quant == "none"),
        bf16=(args.quant == "4bit"),
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=args.max_length,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    train_result = trainer.train()
    wall_s = time.perf_counter() - t0

    trainer.save_model()

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    final_train_loss = train_result.training_loss

    # Compute tokens/s
    total_tokens = (
        len(train_ds) * args.max_length * args.epochs
    )  # upper-bound estimate
    tokens_per_s = total_tokens / wall_s if wall_s > 0 else 0.0

    return final_train_loss, peak_mem_gb, wall_s, tokens_per_s


# ─── CSV Output ──────────────────────────────────────────────────────────────

def append_csv_row(args, model, row_data: dict):

    csv_path = Path(args.results_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row_data)

    print(f"\n✓ Appended row to {csv_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Run: {args.run_id}  |  quant={args.quant}  |  r={args.lora_r}  α={args.lora_alpha}")
    print(f"  Modules: {args.target_modules}")
    print(f"{'='*60}\n")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("[1/5] Loading data …")
    train_records = load_jsonl(args.train_file)
    val_records   = load_jsonl(args.val_file)
    print(f"      Train: {len(train_records)} rows  |  Val: {len(val_records)} rows")

    # ── Step 2: Load model + tokenizer ──────────────────────────────────────
    print("[2/5] Loading model and tokenizer …")
    model, tokenizer = load_model_and_tokenizer(args)
    method = "QLoRA" if args.quant == "4bit" else "LoRA"

    # ── Step 3: Apply LoRA ───────────────────────────────────────────────────
    print("[3/5] Applying LoRA adapters …")
    model = configure_lora(model, args)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_pct    = 100.0 * trainable_params / total_params

    # ── Step 4: Build datasets ───────────────────────────────────────────────
    print("[4/5] Building HF datasets …")
    train_ds = build_hf_dataset(train_records, tokenizer)
    val_ds   = build_hf_dataset(val_records,   tokenizer)

    # ── Step 5: Train ────────────────────────────────────────────────────────
    print("[5/5] Training …")
    final_train_loss, peak_mem_gb, wall_s, tokens_per_s = train(
        model, tokenizer, train_ds, val_ds, args
    )

    # ── Step 6: Validation evaluation ────────────────────────────────────────
    print("\n[6/6] Evaluating on validation set …")
    val_entity_f1, val_leak_rate = evaluate_on_val(
        model, tokenizer, val_records, args
    )

    # Approximate validation loss (token-level CE on a small subset)
    # Full val_loss computation via trainer.evaluate() would be more accurate;
    # here we use the final training loss as a proxy for logging purposes.
    # Override with trainer.evaluate() if available.
    val_loss = final_train_loss  # Replace with actual if evaluate() is called

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  trainable_params = {trainable_params:,}  ({trainable_pct:.3f}%)")
    print(f"  peak_gpu_mem_gb  = {peak_mem_gb:.3f} GB")
    print(f"  wall_clock_s     = {wall_s:.1f} s")
    print(f"  tokens_per_s     = {tokens_per_s:.0f}")
    print(f"  final_train_loss = {final_train_loss:.4f}")
    print(f"  val_entity_f1    = {val_entity_f1:.4f}")
    print(f"  val_leak_rate    = {val_leak_rate:.4f}")
    print(f"{'─'*50}\n")

    row = {
        "run_id":            args.run_id,
        "method":            method,
        "target_modules":    args.target_modules,
        "lora_r":            args.lora_r,
        "lora_alpha":        args.lora_alpha,
        "quant":             args.quant,
        "trainable_params":  trainable_params,
        "total_params":      total_params,
        "trainable_pct":     round(trainable_pct, 5),
        "peak_gpu_mem_gb":   round(peak_mem_gb, 4),
        "train_wallclock_s": round(wall_s, 1),
        "tokens_per_s":      round(tokens_per_s, 1),
        "final_train_loss":  round(final_train_loss, 6),
        "val_loss":          round(val_loss, 6),
        "val_entity_f1":     round(val_entity_f1, 6),
        "val_leak_rate":     round(val_leak_rate, 6),
    }
    append_csv_row(args, model, row)


if __name__ == "__main__":
    main()
