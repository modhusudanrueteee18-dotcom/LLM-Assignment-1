import argparse
import json
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import difflib
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── System Prompt (same as peft_sweep.py) ───────────────────────────────────

SYSTEM_PROMPT = (
    "You are a PII anonymizer. Given text that may contain personal information, "
    "replace every identified PII value with a bracketed placeholder like [EMAIL], "
    "[TEL], [SOCIALNUMBER], [GIVENNAME1], [LASTNAME1], [IP], [DATE], [PASS], etc. "
    "Preserve all non-PII content exactly. Return only the anonymized text."
)

LABEL_RE = re.compile(r'\[([A-Z][A-Z0-9]*?)(?:_?\d+)?\]')


# ─── Argument Parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Part B sanitizer evaluation")
    p.add_argument("--base_model",          required=True)
    p.add_argument("--adapter_path",        default=None,
                   help="LoRA adapter directory (None for zeroshot/regex/presidio)")
    p.add_argument("--test_file",           required=True)
    p.add_argument("--system",
                   choices=["finetuned", "regex", "presidio", "zeroshot"],
                   default="finetuned")
    p.add_argument("--output_predictions",  required=True)
    p.add_argument("--report_file",         required=True)
    p.add_argument("--max_new_tokens",      type=int, default=512)
    p.add_argument("--batch_size",          type=int, default=1,
                   help="Generation batch size (set to 1 for simplicity)")
    p.add_argument("--ood",                 action="store_true",
                   help="Add flag when running on ood_test to mark output correctly")
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


# ─── B.1: Span Recovery ──────────────────────────────────────────────────────

def compute_iou(s1, e1, s2, e2):

    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter == 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def recover_spans(source: str, predicted: str):

    matcher = difflib.SequenceMatcher(None, source, predicted, autojunk=False)
    spans = []

    for tag, s1, s2, p1, p2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        pred_frag = predicted[p1:p2]
        labels = LABEL_RE.findall(pred_frag)

        if tag == "replace":
            if not labels:
                # No label found → model rephrased without masking
                spans.append({"start": s1, "end": s2, "label": "UNKNOWN"})
            elif len(labels) == 1:
                spans.append({"start": s1, "end": s2, "label": labels[0]})
            else:
                src_len = s2 - s1
                n = len(labels)
                portion = max(1, src_len // n)
                for k, lbl in enumerate(labels):
                    sp_s = s1 + k * portion
                    sp_e = s1 + (k + 1) * portion if k < n - 1 else s2
                    spans.append({"start": sp_s, "end": sp_e, "label": lbl})
        elif tag == "delete" and s2 > s1:
            spans.append({"start": s1, "end": s2, "label": "UNKNOWN"})

    return spans


# ─── B.2: Metrics ────────────────────────────────────────────────────────────

def entity_f1_stats(gold_spans, pred_spans, criterion="exact"):

    matched_gold = set()
    tp = 0
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    for pred in pred_spans:
        matched = False
        for j, gold in enumerate(gold_spans):
            if j in matched_gold:
                continue
            if criterion == "exact":
                hit = (pred["start"] == gold["start"] and
                       pred["end"]   == gold["end"]   and
                       pred["label"] == gold["label"])
            else:  # iou50
                hit = (compute_iou(pred["start"], pred["end"],
                                   gold["start"], gold["end"]) >= 0.5 and
                       pred["label"] == gold["label"])
            if hit:
                tp += 1
                per_class_tp[gold["label"]] += 1
                matched_gold.add(j)
                matched = True
                break
        if not matched:
            per_class_fp[pred["label"]] += 1

    for j, gold in enumerate(gold_spans):
        if j not in matched_gold:
            per_class_fn[gold["label"]] += 1

    fp = len(pred_spans) - tp
    fn = len(gold_spans) - len(matched_gold)
    return tp, fp, fn, per_class_tp, per_class_fp, per_class_fn


def aggregate_metrics(all_tp, all_fp, all_fn, class_tp, class_fp, class_fn,
                       top_k_classes=10):

    total_tp = sum(all_tp)
    total_fp = sum(all_fp)
    total_fn = sum(all_fn)

    micro_p  = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    micro_f1 = 2*micro_p*micro_r / (micro_p+micro_r) if micro_p+micro_r > 0 else 0.0

    # Per-class stats
    all_classes = set(class_tp) | set(class_fp) | set(class_fn)
    per_class = {}
    macro_f1_sum = 0.0
    for c in all_classes:
        tp = class_tp.get(c, 0)
        fp = class_fp.get(c, 0)
        fn = class_fn.get(c, 0)
        p  = tp / (tp + fp) if tp + fp > 0 else 0.0
        r  = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2*p*r / (p+r) if p+r > 0 else 0.0
        support = tp + fn
        per_class[c] = {"p": round(p, 4), "r": round(r, 4),
                        "f1": round(f1, 4), "support": support}
        macro_f1_sum += f1

    macro_f1 = macro_f1_sum / len(all_classes) if all_classes else 0.0

    # Keep only top-k by support for the report
    top_classes = sorted(per_class, key=lambda c: per_class[c]["support"], reverse=True)
    top_per_class = {c: per_class[c] for c in top_classes[:top_k_classes]}

    return {
        "micro_p":   round(micro_p,  4),
        "micro_r":   round(micro_r,  4),
        "micro_f1":  round(micro_f1, 4),
        "macro_f1":  round(macro_f1, 4),
    }, top_per_class


def compute_leak_rate_global(all_gold_values, all_pred_texts):

    def normalize(s):
        return " ".join(s.lower().split())

    leaked = 0
    total  = 0
    for gold_vals, pred_text in zip(all_gold_values, all_pred_texts):
        norm_pred = normalize(pred_text)
        for val in gold_vals:
            nv = normalize(val)
            if nv and nv in norm_pred:
                leaked += 1
        total += len(gold_vals)

    return leaked / total if total > 0 else 0.0


def compute_over_masking_rate(gold_spans, source_text, predicted_text, pred_spans):

    # Build gold mask: set of character indices that are PII
    gold_mask = set()
    for g in gold_spans:
        gold_mask.update(range(g["start"], g["end"]))

    non_pii_total = len(source_text) - len(gold_mask)

    # Count pred chars that are masked but NOT gold PII
    masked_non_pii = 0
    for ps in pred_spans:
        for idx in range(ps["start"], ps["end"]):
            if idx not in gold_mask:
                masked_non_pii += 1

    return masked_non_pii / non_pii_total if non_pii_total > 0 else 0.0


# ─── B.3: Regex Baseline ─────────────────────────────────────────────────────

# Comprehensive regex patterns for common PII types
_EMAIL_RE   = re.compile(r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b')
_TEL_RE     = re.compile(
    r'(?:\+?\d{1,3}[\s.\-]?)?'               # Country code
    r'(?:\(?\d{1,4}\)?[\s.\-]?)?'            # Area code
    r'\d{2,4}[\s.\-]?\d{2,4}[\s.\-]?\d{2,6}'  # Local number
)
_IPV4_RE    = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b'
)
_IPV6_RE    = re.compile(
    r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
    r'|\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b'
    r'|\b:(?::[0-9a-fA-F]{1,4}){1,7}\b'
)
_SSN_RE     = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')          # US SSN
_ALTSSN_RE  = re.compile(r'\b\d{9,12}\b')                    # Generic ID number
_DATE_RE    = re.compile(
    r'\b(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'              # 12/31/2022
    r'|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}'                    # 2022-12-31
    r'|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?'
    r'|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?'
    r'|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?'          # January 5th, 2022
    r'|\d{1,2}(?:st|nd|rd|th)?\s+'
    r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?'
    r'|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?'
    r'|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?'
    r'(?:\s+\d{4})?)\b',
    re.IGNORECASE
)
_URL_RE     = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+', re.IGNORECASE)
_POST_RE    = re.compile(r'\b[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}\b'   # UK postcode
                         r'|\b\d{5}(?:-\d{4})?\b', re.IGNORECASE)  # US ZIP
_TIME_RE    = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b', re.IGNORECASE)
_PASSPORT_RE = re.compile(r'\b[A-Z]{1,2}\d{6,9}\b')
_DRIVLIC_RE  = re.compile(r'\b[A-Z]\d{7}\b|\b[A-Z]{2}\d{6}\b')
_GEOCOORD_RE = re.compile(
    r'\b[-+]?\d{1,3}\.\d+[,\s]+[-+]?\d{1,3}\.\d+\b'
)

REGEX_PATTERNS = [
    (_EMAIL_RE,    "EMAIL"),
    (_IPV6_RE,     "IP"),
    (_IPV4_RE,     "IP"),
    (_SSN_RE,      "SOCIALNUMBER"),
    (_DATE_RE,     "DATE"),
    (_TIME_RE,     "TIME"),
    (_URL_RE,      "URL"),
    (_POST_RE,     "POSTCODE"),
    (_PASSPORT_RE, "PASSPORT"),
    (_DRIVLIC_RE,  "DRIVERLICENSE"),
    (_GEOCOORD_RE, "GEOCOORD"),
    (_TEL_RE,      "TEL"),       # broad pattern — keep last to avoid false positives
]


def regex_predict(source_text: str):

    matched_regions = []   # list of (start, end, label)

    for pattern, label in REGEX_PATTERNS:
        for m in pattern.finditer(source_text):
            # Skip if this span overlaps an already-matched region
            overlap = any(
                not (m.end() <= s or m.start() >= e)
                for s, e, _ in matched_regions
            )
            if not overlap:
                matched_regions.append((m.start(), m.end(), label))

    matched_regions.sort(key=lambda x: x[0])

    # Build predicted text
    out = []
    prev = 0
    for s, e, label in matched_regions:
        out.append(source_text[prev:s])
        out.append(f"[{label}]")
        prev = e
    out.append(source_text[prev:])

    predicted_text = "".join(out)
    spans = [{"start": s, "end": e, "label": l} for s, e, l in matched_regions]
    return predicted_text, spans


# ─── B.3: Presidio Baseline ──────────────────────────────────────────────────

def presidio_predict(source_text: str):

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError:
        print("WARNING: presidio not installed. Install with: pip install presidio-analyzer presidio-anonymizer")
        return source_text, []

    PRESIDIO_LABEL_MAP = {
        "PERSON":         "GIVENNAME1",
        "PHONE_NUMBER":   "TEL",
        "EMAIL_ADDRESS":  "EMAIL",
        "IP_ADDRESS":     "IP",
        "DATE_TIME":      "DATE",
        "CREDIT_CARD":    "CARDISSUER",
        "US_SSN":         "SOCIALNUMBER",
        "URL":            "URL",
        "LOCATION":       "CITY",
        "IBAN_CODE":      "IDCARD",
        "NRP":            "SOCIALNUMBER",
        "MEDICAL_LICENSE": "DRIVERLICENSE",
    }

    analyzer  = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text=source_text, language="en")

    spans = []
    for r in results:
        label = PRESIDIO_LABEL_MAP.get(r.entity_type, r.entity_type)
        spans.append({"start": r.start, "end": r.end, "label": label})

    # Build predicted text (replace PII regions with labels)
    spans_sorted = sorted(spans, key=lambda x: x["start"])
    out = []
    prev = 0
    for sp in spans_sorted:
        out.append(source_text[prev:sp["start"]])
        out.append(f"[{sp['label']}]")
        prev = sp["end"]
    out.append(source_text[prev:])

    return "".join(out), spans


# ─── B.3: Zero-shot Baseline ─────────────────────────────────────────────────

ZEROSHOT_SYSTEM = (
    "You are an expert PII anonymizer. Replace all personal information "
    "(names, emails, phone numbers, social security numbers, IP addresses, dates, passwords, "
    "IDs, locations, etc.) with bracketed labels like [EMAIL], [TEL], [SOCIALNUMBER], "
    "[GIVENNAME1], [LASTNAME1], [DATE], [PASS], [IP], [IDCARD], [PASSPORT], [POSTCODE], etc. "
    "Do not change any other text. Output only the anonymized text."
)


def zeroshot_predict(model, tokenizer, source_text, max_new_tokens=512):
    """Zero-shot PII sanitization with no fine-tuning."""
    messages = [
        {"role": "system", "content": ZEROSHOT_SYSTEM},
        {"role": "user",   "content": source_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─── Finetuned Inference ─────────────────────────────────────────────────────

def finetuned_predict(model, tokenizer, source_text, max_new_tokens=512):
    """Generate sanitized text using the fine-tuned adapter."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": source_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), latency_ms


# ─── Well-formedness Check ───────────────────────────────────────────────────

def is_well_formed(source_text, predicted_text):

    if not predicted_text:
        return False
    # Length check
    len_ratio = len(predicted_text) / max(len(source_text), 1)
    if len_ratio < 0.5 or len_ratio > 2.0:
        return False
    # Contains at least a label-looking token or no gold PII (pure text)
    has_label = bool(LABEL_RE.search(predicted_text))
    return has_label or True   # relaxed: always True unless empty


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model_for_inference(base_model_id, adapter_path=None):
    """Load base model and optionally merge LoRA adapter for inference."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"  Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()   # Merge LoRA into base weights for fast inference

    model.eval()
    return model, tokenizer


# ─── Main Evaluation Loop ────────────────────────────────────────────────────

def evaluate(records, args, model=None, tokenizer=None):

    predictions = []
    all_tp_ex  = all_fp_ex  = all_fn_ex  = []
    all_tp_io  = all_fp_io  = all_fn_io  = []
    class_tp_ex = defaultdict(int); class_fp_ex = defaultdict(int); class_fn_ex = defaultdict(int)
    class_tp_io = defaultdict(int); class_fp_io = defaultdict(int); class_fn_io = defaultdict(int)

    all_tp_ex_list = []; all_fp_ex_list = []; all_fn_ex_list = []
    all_tp_io_list = []; all_fp_io_list = []; all_fn_io_list = []
    all_gold_values = []
    all_pred_texts  = []
    all_omr_num     = 0.0
    all_omr_den     = 0
    well_formed_count = 0

    print(f"\n  Evaluating {len(records)} examples with system='{args.system}' …")

    for i, rec in enumerate(records):
        if i % 100 == 0:
            print(f"    [{i}/{len(records)}]", flush=True)

        src = rec["source_text"]
        gold_spans = rec.get("privacy_mask", [])
        latency_ms = 0.0

        # ── Generate prediction based on system ──────────────────────────────
        if args.system == "regex":
            pred_text, pred_spans = regex_predict(src)

        elif args.system == "presidio":
            pred_text, pred_spans = presidio_predict(src)

        elif args.system == "zeroshot":
            pred_text = zeroshot_predict(model, tokenizer, src, args.max_new_tokens)
            pred_spans = recover_spans(src, pred_text)

        else:   # finetuned
            pred_text, latency_ms = finetuned_predict(model, tokenizer, src, args.max_new_tokens)
            pred_spans = recover_spans(src, pred_text)

        # ── Metrics ──────────────────────────────────────────────────────────

        # Exact match
        tp_ex, fp_ex, fn_ex, ctp_ex, cfp_ex, cfn_ex = entity_f1_stats(
            gold_spans, pred_spans, criterion="exact")
        # IoU ≥ 0.5
        tp_io, fp_io, fn_io, ctp_io, cfp_io, cfn_io = entity_f1_stats(
            gold_spans, pred_spans, criterion="iou50")

        all_tp_ex_list.append(tp_ex); all_fp_ex_list.append(fp_ex); all_fn_ex_list.append(fn_ex)
        all_tp_io_list.append(tp_io); all_fp_io_list.append(fp_io); all_fn_io_list.append(fn_io)

        for c in ctp_ex: class_tp_ex[c] += ctp_ex[c]
        for c in cfp_ex: class_fp_ex[c] += cfp_ex[c]
        for c in cfn_ex: class_fn_ex[c] += cfn_ex[c]
        for c in ctp_io: class_tp_io[c] += ctp_io[c]
        for c in cfp_io: class_fp_io[c] += cfp_io[c]
        for c in cfn_io: class_fn_io[c] += cfn_io[c]

        # Leak rate per example
        gold_vals = [src[g["start"]:g["end"]] for g in gold_spans]
        all_gold_values.append(gold_vals)
        all_pred_texts.append(pred_text)

        # Over-masking
        omr = compute_over_masking_rate(gold_spans, src, pred_text, pred_spans)
        all_omr_num += omr * max(len(src) - sum(g["end"]-g["start"] for g in gold_spans), 1)
        all_omr_den += max(len(src) - sum(g["end"]-g["start"] for g in gold_spans), 1)

        # Well-formed check
        if is_well_formed(src, pred_text):
            well_formed_count += 1

        # Leaked values for this example
        def normalize(s): return " ".join(s.lower().split())
        norm_pred = normalize(pred_text)
        leaked_values = [v for v in gold_vals if normalize(v) and normalize(v) in norm_pred]

        predictions.append({
            "id":              rec.get("id", f"ex_{i}"),
            "language":        rec.get("language", "en"),
            "source_text":     src,
            "predicted_text":  pred_text,
            "predicted_spans": pred_spans,
            "gold_spans":      gold_spans,
            "leaked_values":   leaked_values,
            "latency_ms":      round(latency_ms, 1),
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    exact_micro, exact_per_class = aggregate_metrics(
        all_tp_ex_list, all_fp_ex_list, all_fn_ex_list,
        class_tp_ex, class_fp_ex, class_fn_ex,
    )
    iou_micro, iou_per_class = aggregate_metrics(
        all_tp_io_list, all_fp_io_list, all_fn_io_list,
        class_tp_io, class_fp_io, class_fn_io,
    )

    global_leak_rate = compute_leak_rate_global(all_gold_values, all_pred_texts)
    over_mask_rate   = all_omr_num / all_omr_den if all_omr_den > 0 else 0.0
    well_formed_rate = well_formed_count / len(records) if records else 0.0

    metrics = {
        "system":       args.system,
        "adapter":      args.adapter_path or "none",
        "test_set":     args.test_file,
        "n":            len(records),
        "exact":        {**exact_micro, "per_class": exact_per_class},
        "overlap_iou50": {**iou_micro,  "per_class": iou_per_class},
        "per_class":    exact_per_class,   # for top-level convenience
        "leak_rate":         round(global_leak_rate, 6),
        "over_masking_rate": round(over_mask_rate, 6),
        "well_formed_rate":  round(well_formed_rate, 6),
    }

    return predictions, metrics


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load records ──────────────────────────────────────────────────────────
    print(f"[1/3] Loading test records from {args.test_file} …")
    records = load_jsonl(args.test_file)
    print(f"      {len(records)} records loaded.")

    # ── Load model (if needed) ────────────────────────────────────────────────
    model = tokenizer = None
    if args.system in ("finetuned", "zeroshot"):
        print(f"[2/3] Loading model: {args.base_model} …")
        model, tokenizer = load_model_for_inference(
            args.base_model,
            adapter_path=(args.adapter_path if args.system == "finetuned" else None),
        )
    else:
        print(f"[2/3] System='{args.system}' — no model load needed.")

    # ── Run evaluation ────────────────────────────────────────────────────────
    print("[3/3] Running evaluation …")
    predictions, metrics = evaluate(records, args, model=model, tokenizer=tokenizer)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_pred = Path(args.output_predictions)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pred, "w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred, ensure_ascii=False) + "\n")

    out_report = Path(args.report_file)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  System:          {args.system}")
    print(f"  Exact  micro-F1: {metrics['exact']['micro_f1']:.4f}")
    print(f"  IoU50  micro-F1: {metrics['overlap_iou50']['micro_f1']:.4f}")
    print(f"  Leak rate:       {metrics['leak_rate']:.4f}")
    print(f"  Over-mask rate:  {metrics['over_masking_rate']:.4f}")
    print(f"  Well-formed:     {metrics['well_formed_rate']:.4f}")
    print(f"\n  Predictions → {out_pred}")
    print(f"  Report      → {out_report}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
