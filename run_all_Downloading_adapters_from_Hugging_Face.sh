#!/usr/bin/env bash

set -e   # Exit immediately on any error
set -u   # Treat unset variables as errors

SEED=42
MODEL_SANITIZER="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_VICTIM="Qwen/Qwen2.5-0.5B"

DATA_DIR="data"
RESULTS_DIR="results"
SRC_DIR="src"

# --- ADAPTER PATHS ---
# These point to the local directory after downloading from your Hugging Face repository
HF_SANITIZER_ADAPTER="hf_download/adapters/best_sanitizer"
HF_VICTIM_RAW_ADAPTER="hf_download/adapters/victim_raw"
HF_VICTIM_SAN_ADAPTER="hf_download/adapters/victim"

mkdir -p "$RESULTS_DIR"

# Step 0: Integrity check
echo " Step 0: Verifying dataset integrity"

python - <<'PY'
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

# Step 0.5: Automatic Hugging Face Download
echo " Step 0.5: Downloading adapters from Hugging Face"
hf download Modhusudan/Assignment1-lora-adapters --local-dir hf_download

echo ""
echo " Part A: PEFT Sweep (SKIPPED FOR EVALUATION)"
echo ""
echo " Part B: Sanitizer Evaluation"

# B.1–B.3: Finetuned system on partB_test.jsonl using downloaded Hugging Face weights
echo "[B] Finetuned sanitizer on in-distribution test set …"
python $SRC_DIR/sanitize_eval.py \
  --base_model $MODEL_SANITIZER \
  --adapter_path $HF_SANITIZER_ADAPTER \
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

# B.4: OOD evaluation (finetuned) using downloaded Hugging Face weights
echo "[B.4] OOD evaluation …"
python $SRC_DIR/sanitize_eval.py \
  --base_model $MODEL_SANITIZER \
  --adapter_path $HF_SANITIZER_ADAPTER \
  --test_file $DATA_DIR/partB_ood_test.jsonl \
  --system finetuned --ood \
  --output_predictions $RESULTS_DIR/partB_predictions_ood.jsonl \
  --report_file $RESULTS_DIR/partB_metrics_ood.json

echo "Part B complete."

echo ""
echo " Part C: Victim Model Training (SKIPPED FOR EVALUATION)"
# C.2a and C.2c local training steps are skipped.

# C.2b: Create sanitized corpus using downloaded Hugging Face weights
echo "[C] Creating sanitized corpus …"
python - <<EOF
import sys
sys.path.insert(0, "src")
import json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_victim import sanitize_corpus

base_model = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "${HF_SANITIZER_ADAPTER}"

tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

mdl = PeftModel.from_pretrained(mdl, adapter_path).merge_and_unload()
mdl.eval()

sanitize_corpus(
    "data/partC_corpus_raw.jsonl",
    mdl, tok,
    "data/partC_corpus_sanitized.jsonl"
)
EOF

echo ""
echo " Part C: Extraction Attacks"

# C.3: Attacks on V-raw using downloaded Hugging Face weights
echo "[C] Running all attacks on V-raw …"
python $SRC_DIR/attack_extract.py \
  --victim_model $MODEL_VICTIM \
  --adapter_path $HF_VICTIM_RAW_ADAPTER \
  --ref_model $MODEL_VICTIM \
  --corpus $DATA_DIR/partC_corpus_raw.jsonl \
  --canaries $DATA_DIR/partC_canaries.jsonl \
  --targeted_file $DATA_DIR/partC_targeted.jsonl \
  --attack all \
  --num_samples 5000 --seed $SEED \
  --victim_name victim_raw \
  --report_file $RESULTS_DIR/partC_attacks_raw.json

# C.3: Attacks on V-san using downloaded Hugging Face weights
echo "[C] Running all attacks on V-san …"
python $SRC_DIR/attack_extract.py \
  --victim_model $MODEL_VICTIM \
  --adapter_path $HF_VICTIM_SAN_ADAPTER \
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
echo " All evaluation steps complete. Results in: $RESULTS_DIR/"
ls -lh $RESULTS_DIR/