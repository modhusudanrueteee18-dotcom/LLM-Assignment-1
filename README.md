# requirements.txt — Assignment 1, ELL8299/COL8383/COL873
# Install: pip install -r requirements.txt
# Python 3.10+, CUDA 11.8+

# ── Core deep-learning stack ───────────────────────────────────────────────
torch>=2.1.0                 # https://pytorch.org/
torchvision>=0.16.0

# ── HuggingFace ecosystem ──────────────────────────────────────────────────
transformers>=4.45.0         # https://github.com/huggingface/transformers
datasets>=2.20.0             # https://github.com/huggingface/datasets
tokenizers>=0.19.0
accelerate>=0.32.0           # https://github.com/huggingface/accelerate
peft>=0.12.0                 # https://github.com/huggingface/peft
trl>=0.9.6                   # https://github.com/huggingface/trl (SFTTrainer)
bitsandbytes>=0.43.0         # NF4 quantization for QLoRA

# ── PII baselines ──────────────────────────────────────────────────────────
presidio-analyzer>=2.2.355   # https://microsoft.github.io/presidio/
presidio-anonymizer>=2.2.355
spacy>=3.7.0                 # Presidio NLP backend

# ── Data + numerics ───────────────────────────────────────────────────────
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.4.0          # For AUC-ROC in membership inference

# ── Plotting ──────────────────────────────────────────────────────────────
matplotlib>=3.8.0
seaborn>=0.13.0

# ── Utilities ─────────────────────────────────────────────────────────────
tqdm>=4.66.0
regex>=2024.0.0
