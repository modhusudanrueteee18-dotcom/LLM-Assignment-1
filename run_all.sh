#!/usr/bin/env bash
# =============================================================================
# run_all.sh  —  End-to-End Pipeline for Assignment 1
# ELL8299/COL8383/COL873: Trustworthy Large Language Models, Sem I 2026-27
# =============================================================================
# Prerequisites:
#   pip install -r requirements.txt
#   GPU with ≥16 GB VRAM (V100 reference environment)
#   All data files verified via manifest.jsonl integrity check
#
# Expected total wall-clock: ~6–10 hours on a single V100 (all 6 LoRA runs +
# sanitization eval + victim training × 2 + extraction attacks × 2).
# =============================================================================

set -e   # Exit immediately on any error
set -u   # Treat unset variables as errors

SEED=42
MODEL_SANITIZER="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_VICTIM="Qwen/Qwen2.5-0.5B"

DATA_DIR="data"
ADAPTER_DIR="adapters"
RESULTS_DIR="results"
SRC_DIR="src"

mkdir -p "$ADAPTER_DIR" "$RESULTS_DIR"

# =============================================================================
# Step 0: Integrity check
# =============================================================================
echo "========================================="
echo " Step 0: Verifying dataset integrity"
echo "========================================="
python3 - <<'PY'
import hashlib, json, pathlib
d = pathlib.Path("data")
for line in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    payload = (d / item["file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == item["sha256"], \
        f"SHA-256 mismatch for {item['file']}"
    assert len(payload.splitlines()) == item["rows"], \
        f"Row count mismatch for {item['file']}"
print("dataset bundle OK")
PY

# =============================================================================
# Part A — PEFT Sweep (6 configurations)
# =============================================================================
echo ""
echo "========================================="
echo " Part A: PEFT Sweep"
echo "========================================="

EPOCHS=1
BATCH=4
GRAD_ACCUM=4
LR=2e-4
CSV="$RESULTS_DIR/partA_sweep.csv"

# --- A1: LoRA | q_proj, v_proj | r=8, α=16 ---
echo "[A1] LoRA q+v  r=8  α=16"
python $SRC_DIR/peft_sweep.py \
  --model $MODEL_SANITIZER \
  --train_file $DATA_DIR/partA_train.jsonl \
  --val_file   $DATA_DIR/partA_validation.jsonl \
  --target_modules q_proj,v_proj \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --quant none --seed $SEED \
  --run_id A1 --output_dir $ADAPTER_DIR/A1 \
  --results_csv $CSV \
  --epochs $EPOCHS --per_device_batch $BATCH --grad_accum $GRAD_ACCUM --lr $LR

# --- A2: LoRA | q, k, v, o | r=8, α=16 ---
echo "[A2] LoRA q+k+v+o  r=8  α=16"
python $SRC_DIR/peft_sweep.py \
  --model $MODEL_SANITIZER \
  --train_file $DATA_DIR/partA_train.jsonl \
  --val_file   $DATA_DIR/partA_validation.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --quant none --seed $SEED \
  --run_id A2 --output_dir $ADAPTER_DIR/A2 \
  --results_csv $CSV \
  --epochs $EPOCHS --per_device_batch $BATCH --grad_accum $GRAD_ACCUM --lr $LR

# --- A3: LoRA | attn + MLP | r=8, α=16 ---
echo "[A3] LoRA attn+MLP  r=8  α=16"
python $SRC_DIR/peft_sweep.py \
  --model $MODEL_SANITIZER \
  --train_file $DATA_DIR/partA_train.jsonl \
  --val_file   $DATA_DIR/partA_validation.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --quant none --seed $SEED \
  --run_id A3 --output_dir $ADAPTER_DIR/A3 \
  --results_csv $CSV \
  --epochs $EPOCHS --per_device_batch $BATCH --grad_accum $GRAD_ACCUM --lr $LR

# --- A4: LoRA | attn + MLP | r=32, α=16 ---
echo "[A4] LoRA attn+MLP  r=32  α=16"
python $SRC_DIR/peft_sweep.py \
  --model $MODEL_SANITIZER \
  --train_file $DATA_DIR/partA_train.jsonl \
  --val_file   $DATA_DIR/partA_validation.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 16 --lora_dropout 0.05 \
  --quant none --seed $SEED \
  --run_id A4 --output_dir $ADAPTER_DIR/A4 \
  --results_csv $CSV \
  --epochs $EPOCHS --per_device_batch $BATCH --grad_accum $GRAD_ACCUM --lr $LR

# --- A5: LoRA | attn + MLP | r=32, α=64 ---
echo "[A5] LoRA attn+MLP  r=32  α=64"
python $SRC_DIR/peft_sweep.py \
  --model $MODEL_SANITIZER \
  --train_file $DATA_DIR/partA_train.jsonl \
  --val_file   $DATA_DIR/partA_validation.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --quant none --seed $SEED \
  --run_id A5 --output_dir $ADAPTER_DIR/A5 \
  --results_csv $CSV \
  --epochs $EPOCHS --per_device_batch $BATCH --grad_accum $GRAD_ACCUM --lr $LR

# --- A6: QLoRA | attn + MLP | r=32, α=64 ---
echo "[A6] QLoRA attn+MLP  r=32  α=64"
python $SRC_DIR/peft_sweep.py \
  --model $MODEL_SANITIZER \
  --train_file $DATA_DIR/partA_train.jsonl \
  --val_file   $DATA_DIR/partA_validation.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --quant 4bit --seed $SEED \
  --run_id A6 --output_dir $ADAPTER_DIR/A6 \
  --results_csv $CSV \
  --epochs $EPOCHS --per_device_batch $BATCH --grad_accum $GRAD_ACCUM --lr $LR

# Generate Part A figure
echo "[A] Generating Part A figure …"
python $SRC_DIR/plot_sweep.py \
  --csv $CSV \
  --out $RESULTS_DIR/partA_fig.png

echo "Part A complete. Sweep results: $CSV"

# =============================================================================
# Part B — Sanitizer Evaluation
# Select best adapter (justify in report; default: A5 — highest F1/param)
# =============================================================================
echo ""
echo "========================================="
echo " Part B: Sanitizer Evaluation"
echo "========================================="

# Copy best adapter — change A5 to whichever run_id you select
BEST_ADAPTER="$ADAPTER_DIR/A5"
cp -r "$BEST_ADAPTER" "$ADAPTER_DIR/best_sanitizer" 2>/dev/null || true

# B.1–B.3: Finetuned system on partB_test.jsonl
echo "[B] Finetuned sanitizer on in-distribution test set …"
python $SRC_DIR/sanitize_eval.py \
  --base_model $MODEL_SANITIZER \
  --adapter_path $ADAPTER_DIR/best_sanitizer \
  --test_file $DATA_DIR/partB_test.jsonl \
  --system finetuned \
  --output_predictions $RESULTS_DIR/partB_predictions.jsonl \
  --report_file $RESULTS_DIR/partB_metrics.json

# Regex baseline
echo "[B] Regex baseline …"
python $SRC_DIR/sanitize_eval.py \
  --base_model $MODEL_SANITIZER \
  --test_file $DATA_DIR/partB_test.jsonl \
  --system regex \
  --output_predictions $RESULTS_DIR/partB_predictions_regex.jsonl \
  --report_file $RESULTS_DIR/partB_metrics_regex.json

# Presidio baseline
echo "[B] Presidio baseline …"
python $SRC_DIR/sanitize_eval.py \
  --base_model $MODEL_SANITIZER \
  --test_file $DATA_DIR/partB_test.jsonl \
  --system presidio \
  --output_predictions $RESULTS_DIR/partB_predictions_presidio.jsonl \
  --report_file $RESULTS_DIR/partB_metrics_presidio.json

# Zero-shot baseline
echo "[B] Zero-shot baseline …"
python $SRC_DIR/sanitize_eval.py \
  --base_model $MODEL_SANITIZER \
  --test_file $DATA_DIR/partB_test.jsonl \
  --system zeroshot \
  --output_predictions $RESULTS_DIR/partB_predictions_zeroshot.jsonl \
  --report_file $RESULTS_DIR/partB_metrics_zeroshot.json

# B.4: OOD evaluation (finetuned)
echo "[B.4] OOD evaluation …"
python $SRC_DIR/sanitize_eval.py \
  --base_model $MODEL_SANITIZER \
  --adapter_path $ADAPTER_DIR/best_sanitizer \
  --test_file $DATA_DIR/partB_ood_test.jsonl \
  --system finetuned --ood \
  --output_predictions $RESULTS_DIR/partB_predictions_ood.jsonl \
  --report_file $RESULTS_DIR/partB_metrics_ood.json

echo "Part B complete."

# =============================================================================
# Part C — Victim Training + Attacks
# =============================================================================
echo ""
echo "========================================="
echo " Part C: Victim Model Training"
echo "========================================="

VICTIM_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# C.2a: Train V-raw
echo "[C] Training V-raw …"
python $SRC_DIR/train_victim.py \
  --model $MODEL_VICTIM \
  --corpus $DATA_DIR/partC_corpus_raw.jsonl \
  --heldout $DATA_DIR/partC_heldout.jsonl \
  --target_modules $VICTIM_MODULES \
  --lora_r 32 --lora_alpha 64 --epochs 4 --seed $SEED \
  --output_dir $ADAPTER_DIR/victim_raw \
  --report_json $RESULTS_DIR/victim_raw_training.json

# C.2b: Create sanitized corpus
echo "[C] Creating sanitized corpus …"
python3 - <<'PY'
import sys
sys.path.insert(0, "src")
import json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_victim import sanitize_corpus

base_model = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "adapters/best_sanitizer"

tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
from peft import PeftModel
mdl = PeftModel.from_pretrained(mdl, adapter_path).merge_and_unload()
mdl.eval()

sanitize_corpus(
    "data/partC_corpus_raw.jsonl",
    mdl, tok,
    "data/partC_corpus_sanitized.jsonl"
)
PY

# C.2c: Train V-san
echo "[C] Training V-san …"
python $SRC_DIR/train_victim.py \
  --model $MODEL_VICTIM \
  --corpus $DATA_DIR/partC_corpus_sanitized.jsonl \
  --heldout $DATA_DIR/partC_heldout.jsonl \
  --target_modules $VICTIM_MODULES \
  --lora_r 32 --lora_alpha 64 --epochs 4 --seed $SEED \
  --output_dir $ADAPTER_DIR/victim_san \
  --report_json $RESULTS_DIR/victim_san_training.json

echo ""
echo "========================================="
echo " Part C: Extraction Attacks"
echo "========================================="

# C.3: Attacks on V-raw
echo "[C] Running all attacks on V-raw …"
python $SRC_DIR/attack_extract.py \
  --victim_model $MODEL_VICTIM \
  --adapter_path $ADAPTER_DIR/victim_raw \
  --ref_model $MODEL_VICTIM \
  --corpus $DATA_DIR/partC_corpus_raw.jsonl \
  --canaries $DATA_DIR/partC_canaries.jsonl \
  --targeted_file $DATA_DIR/partC_targeted.jsonl \
  --attack all \
  --num_samples 5000 --seed $SEED \
  --victim_name victim_raw \
  --report_file $RESULTS_DIR/partC_attacks_raw.json

# C.3: Attacks on V-san
echo "[C] Running all attacks on V-san …"
python $SRC_DIR/attack_extract.py \
  --victim_model $MODEL_VICTIM \
  --adapter_path $ADAPTER_DIR/victim_san \
  --ref_model $MODEL_VICTIM \
  --corpus $DATA_DIR/partC_corpus_sanitized.jsonl \
  --canaries $DATA_DIR/partC_canaries.jsonl \
  --targeted_file $DATA_DIR/partC_targeted.jsonl \
  --attack all \
  --num_samples 5000 --seed $SEED \
  --victim_name victim_san \
  --report_file $RESULTS_DIR/partC_attacks_san.json

# Generate exposure plot
echo "[C] Generating exposure figure …"
python $SRC_DIR/plot_exposure.py \
  --raw_report $RESULTS_DIR/partC_attacks_raw.json \
  --san_report $RESULTS_DIR/partC_attacks_san.json \
  --out $RESULTS_DIR/partC_exposure.png

echo ""
echo "========================================="
echo " All steps complete. Results in: $RESULTS_DIR/"
echo "========================================="
ls -lh $RESULTS_DIR/
