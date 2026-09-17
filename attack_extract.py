import argparse
import json
import math
import random
import re
import string
import zlib
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Argument Parsing ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Part C extraction attacks")
    p.add_argument("--victim_model",  required=True, help="Base model ID for victim")
    p.add_argument("--adapter_path",  required=True, help="LoRA adapter path for victim")
    p.add_argument("--ref_model",     required=True, help="Pre-trained reference model (no fine-tuning)")
    p.add_argument("--corpus",        required=True, help="partC_corpus_raw.jsonl or sanitized")
    p.add_argument("--canaries",      required=True, help="partC_canaries.jsonl")
    p.add_argument("--targeted_file", required=True, help="partC_targeted.jsonl")
    p.add_argument("--attack",
                   choices=["discoverable", "untargeted", "canary", "targeted", "all"],
                   default="all")
    p.add_argument("--num_samples",   type=int, default=5000,
                   help="Number of untargeted samples to generate")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--report_file",   required=True)
    p.add_argument("--max_new_tokens",type=int, default=200,
                   help="Max tokens for untargeted/targeted generation")
    p.add_argument("--prefix_len",    type=int, default=50,
                   help="Number of tokens to use as prefix for discoverable attack")
    p.add_argument("--n_canary_alts", type=int, default=9999,
                   help="Number of alternative secrets for canary ranking")
    p.add_argument("--victim_name",   default="victim_raw",
                   help="Label for output JSON (victim_raw or victim_san)")
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


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model(model_id, adapter_path=None):
    """Load model + optional LoRA adapter. Merge for inference speed."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    if adapter_path and Path(adapter_path).exists():
        mdl = PeftModel.from_pretrained(mdl, adapter_path)
        mdl = mdl.merge_and_unload()

    mdl.eval()
    return mdl, tok


# ─── Log-Probability Utilities ───────────────────────────────────────────────

def compute_log_prob(model, tokenizer, text, max_length=512, device=None):

    if device is None:
        device = next(model.parameters()).device

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids.to(device)
    n = input_ids.shape[1]
    if n < 2:
        return 0.0, 0

    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
        # out.loss = mean cross-entropy = mean NLL per token
        mean_nll = out.loss.item()
        total_nll = mean_nll * (n - 1)   # approximate (ignores first token)

    total_log_prob = -total_nll
    return total_log_prob, n - 1


def compute_nll(model, tokenizer, text, max_length=512):

    lp, n = compute_log_prob(model, tokenizer, text, max_length)
    if n == 0:
        return float("inf")
    return -lp / n   # mean NLL per token = log-ppl (in nats)


def zlib_entropy(text: str) -> float:

    return len(zlib.compress(text.encode("utf-8"), 9))


# ─── Membership Scoring ──────────────────────────────────────────────────────

def membership_scores(victim, victim_tok, ref, ref_tok, text, max_length=512):

    nll_v = compute_nll(victim, victim_tok, text, max_length)
    nll_r = compute_nll(ref,    ref_tok,    text, max_length)
    nll_lower = compute_nll(victim, victim_tok, text.lower(), max_length)

    zlib_bytes = zlib_entropy(text)
    n_bytes    = len(text.encode("utf-8"))

    ratio_ref   = nll_v / nll_r          if nll_r > 0 else float("inf")
    ratio_lower = nll_v / nll_lower      if nll_lower > 0 else float("inf")
    ratio_zlib  = nll_v / (zlib_bytes / max(n_bytes, 1)) if zlib_bytes > 0 else float("inf")

    return {
        "text":         text,
        "ppl":          nll_v,
        "ratio_ref":    ratio_ref,
        "ratio_lower":  ratio_lower,
        "ratio_zlib":   ratio_zlib,
    }


# ─── PII Pattern (for PII recovery in untargeted attack) ────────────────────

PII_RE = re.compile(
    r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'          # EMAIL
    r'|\b\d{3}-\d{2}-\d{4}\b'                                           # SSN
    r'|\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b'                           # IPv4
    r'|\+?\d[\d\s.\-()]{7,14}\d'                                        # PHONE
    r'|\b[A-Z]{1,2}\d{6,9}\b'                                           # PASSPORT
)

# Known PII values in corpus (loaded at runtime)
CORPUS_PII_VALUES = set()


def load_corpus_pii(corpus_records):
    """Extract PII-looking strings from corpus for PII recovery check."""
    for rec in corpus_records:
        for m in PII_RE.finditer(rec.get("text", "")):
            CORPUS_PII_VALUES.add(m.group().lower().strip())


def count_pii_strings(text):
    """Count PII-looking substrings in a generated sample that appear in the training corpus."""
    count = 0
    for m in PII_RE.finditer(text):
        val = m.group().lower().strip()
        if val in CORPUS_PII_VALUES:
            count += 1
    return count


# ─── (a) Discoverable Memorization ──────────────────────────────────────────

def attack_discoverable(victim, victim_tok, corpus_records, prefix_len=50,
                         max_new_tokens=200, seed=42):

    random.seed(seed)

    # Group records by n_reps
    by_reps = defaultdict(list)
    for rec in corpus_records:
        by_reps[rec["n_reps"]].append(rec)

    device = next(victim.parameters()).device
    results_by_reps = {}

    print("  [Discoverable] Running greedy prefix attack …")

    for n_reps, recs in sorted(by_reps.items()):
        # Sample up to 100 per bucket to keep runtime manageable
        sample_recs = random.sample(recs, min(100, len(recs)))

        exact_matches  = 0
        edit_sims      = []

        for rec in sample_recs:
            text = rec["text"]
            enc  = victim_tok(text, return_tensors="pt", truncation=True,
                              max_length=512).input_ids.to(device)

            if enc.shape[1] <= prefix_len:
                continue   # too short to split

            # Prefix: first `prefix_len` tokens
            prefix_ids = enc[:, :prefix_len]
            gt_suffix_ids = enc[:, prefix_len:]

            with torch.no_grad():
                gen_ids = victim.generate(
                    prefix_ids,
                    max_new_tokens=min(max_new_tokens, gt_suffix_ids.shape[1]),
                    do_sample=False,                # greedy decoding
                    pad_token_id=victim_tok.eos_token_id,
                )
            gen_suffix_ids = gen_ids[:, prefix_len:]

            # Decode both suffixes for comparison
            gt_text  = victim_tok.decode(gt_suffix_ids[0],  skip_special_tokens=True)
            gen_text = victim_tok.decode(gen_suffix_ids[0], skip_special_tokens=True)

            # Exact match (token-level, truncated to gt length)
            compare_len = min(gt_suffix_ids.shape[1], gen_suffix_ids.shape[1])
            exact = (gt_suffix_ids[0, :compare_len] == gen_suffix_ids[0, :compare_len]).all().item()
            exact_matches += int(exact)

            # Edit similarity
            sim = edit_similarity(gt_text, gen_text)
            edit_sims.append(sim)

        n = len(sample_recs)
        em_rate  = exact_matches / n if n > 0 else 0.0
        mean_sim = sum(edit_sims) / len(edit_sims) if edit_sims else 0.0

        results_by_reps[str(n_reps)] = {
            "n_sampled":       n,
            "exact_match_rate": round(em_rate, 4),
            "mean_edit_sim":   round(mean_sim, 4),
        }
        print(f"    n_reps={n_reps:3d}: EM={em_rate:.3f}, EditSim={mean_sim:.3f} (n={n})")

    overall_em  = sum(v["exact_match_rate"] * v["n_sampled"]
                      for v in results_by_reps.values()) / max(
                      sum(v["n_sampled"] for v in results_by_reps.values()), 1)
    overall_sim = sum(v["mean_edit_sim"] * v["n_sampled"]
                      for v in results_by_reps.values()) / max(
                      sum(v["n_sampled"] for v in results_by_reps.values()), 1)

    return {
        "exact_match_rate":  round(overall_em,  4),
        "mean_edit_sim":     round(overall_sim, 4),
        "by_n_reps":         results_by_reps,
    }


def edit_similarity(a: str, b: str) -> float:

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Truncate to 500 chars for speed
    a, b = a[:500], b[:500]
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    ed = dp[n]
    return 1.0 - ed / max(m, n)


# ─── (b) Untargeted Extraction ───────────────────────────────────────────────

def attack_untargeted(victim, victim_tok, ref, ref_tok, corpus_records,
                       num_samples=5000, max_new_tokens=200, seed=42):

    random.seed(seed)
    device = next(victim.parameters()).device

    # Collect all corpus texts for verbatim checking
    corpus_texts = [r["text"] for r in corpus_records]
    corpus_text_lower = [t.lower() for t in corpus_texts]

    # Build short prompts from random corpus prefixes
    all_prompts = []
    for rec in corpus_records:
        tokens = victim_tok(rec["text"], truncation=True, max_length=200).input_ids
        if len(tokens) >= 6:
            # Use a random 3–10 token prefix
            n_prefix = random.randint(3, min(10, len(tokens) // 2))
            prefix_ids = tokens[:n_prefix]
            all_prompts.append(prefix_ids)
    random.shuffle(all_prompts)

    # Also add fully random prompts (single BOS token) for diversity
    n_random_prompts = num_samples // 10
    for _ in range(n_random_prompts):
        all_prompts.append([victim_tok.bos_token_id or 1])

    print(f"  [Untargeted] Generating {num_samples} samples …")

    generated_texts = []
    i = 0
    while len(generated_texts) < num_samples:
        prompt_ids = all_prompts[i % len(all_prompts)]
        i += 1

        prompt_tensor = torch.tensor([prompt_ids]).to(device)

        with torch.no_grad():
            gen_ids = victim.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=40,
                pad_token_id=victim_tok.eos_token_id,
            )
        gen_text = victim_tok.decode(gen_ids[0], skip_special_tokens=True)
        if len(gen_text.strip()) > 20:   # skip degenerate outputs
            generated_texts.append(gen_text)

        if len(generated_texts) % 500 == 0:
            print(f"    Generated {len(generated_texts)}/{num_samples}", flush=True)

    # Deduplicate
    unique_texts  = list(dict.fromkeys(generated_texts))
    n_unique      = len(unique_texts)
    print(f"    {n_unique} unique samples (from {len(generated_texts)} total)")

    # Score all samples
    print("  [Untargeted] Scoring samples …")
    scored = []
    for k, text in enumerate(unique_texts):
        if k % 500 == 0:
            print(f"    Scoring {k}/{n_unique}", flush=True)
        scores = membership_scores(victim, victim_tok, ref, ref_tok, text)
        scored.append(scores)

    # Helper: check if a generated text appears verbatim in training corpus
    def is_verbatim_training(text, min_len=20):
        t = text.lower()
        return any(t[pos:pos+min_len] in ct
                   for pos in range(0, len(t)-min_len, min_len//2)
                   for ct in corpus_text_lower)

    # Compute Precision@100 for each ranker
    def precision_at_k(scored_texts, score_key, k=100):
        """
        Sort by `score_key` (ascending) and compute fraction of top-k
        that contain verbatim training data.
        """
        ranked = sorted(scored_texts, key=lambda x: x[score_key])
        top_k  = ranked[:k]
        hits   = sum(1 for x in top_k if is_verbatim_training(x["text"]))
        pii_count = sum(count_pii_strings(x["text"]) for x in top_k)

        # Distinct training documents recovered
        distinct_docs = set()
        for x in top_k:
            t = x["text"].lower()
            for ct_orig, ct_lower in zip(corpus_texts, corpus_text_lower):
                if any(t[pos:pos+20] in ct_lower
                       for pos in range(0, max(len(t)-20, 1), 10)):
                    distinct_docs.add(ct_orig[:30])  # use first 30 chars as doc ID

        return {
            score_key:         round(hits / k, 4),
            "verified_pii_strings": pii_count,
            "distinct_docs_recovered": len(distinct_docs),
        }

    p100_ppl   = precision_at_k(scored, "ppl")
    p100_ref   = precision_at_k(scored, "ratio_ref")
    p100_lower = precision_at_k(scored, "ratio_lower")
    p100_zlib  = precision_at_k(scored, "ratio_zlib")

    return {
        "n_generated":  len(generated_texts),
        "n_unique":     n_unique,
        "precision_at_100": {
            "ppl":          p100_ppl["ppl"],
            "ratio_ref":    p100_ref["ratio_ref"],
            "ratio_lower":  p100_lower["ratio_lower"],
            "ratio_zlib":   p100_zlib["ratio_zlib"],
        },
        "distinct_docs_recovered": p100_ppl["distinct_docs_recovered"],
        "verified_pii_strings":    p100_ppl["verified_pii_strings"],
    }


# ─── (c) Canary Exposure ─────────────────────────────────────────────────────

def generate_6digit_alternatives(true_secret: str, n: int, rng: random.Random):

    alts = set()
    while len(alts) < n:
        s = f"{rng.randint(0, 999999):06d}"
        if s != true_secret:
            alts.add(s)
    return list(alts)


def score_canary_candidate(model, tokenizer, template: str, project: str,
                            secret: str, max_length=128):

    text = template.format(project=project, secret=secret)
    nll  = compute_nll(model, tokenizer, text, max_length)
    return nll


def attack_canary_exposure(victim, victim_tok, canary_records,
                            n_alternatives=9999, seed=42):

    rng = random.Random(seed)
    R = 10 ** 6    # |R|: number of possible 6-digit secrets

    # Group canaries by n_reps
    by_reps = defaultdict(list)
    for c in canary_records:
        by_reps[c["n_reps"]].append(c)

    results_by_reps = {}

    print("  [Canary] Computing exposure …")

    for n_reps, canaries in sorted(by_reps.items()):
        exposures = []

        for canary in canaries:
            true_secret = canary["secret"]
            project     = canary["project"]
            template    = canary["template"]

            # Generate alternatives
            alternatives = generate_6digit_alternatives(true_secret, n_alternatives, rng)
            all_candidates = [true_secret] + alternatives  # true secret at index 0

            # Score all candidates
            nlls = []
            for secret in all_candidates:
                nll = score_canary_candidate(victim, victim_tok, template, project, secret)
                nlls.append(nll)

            # Rank by NLL (ascending: lower NLL = higher prob = better rank)
            sorted_idx = sorted(range(len(nlls)), key=lambda i: nlls[i])
            rank_in_sample = sorted_idx.index(0) + 1    # 1-indexed rank of true secret

            # Sampled-rank estimator
            rank_hat = (rank_in_sample / len(all_candidates)) * R

            # Exposure formula (Carlini et al., 2019)
            exposure = math.log2(R) - math.log2(rank_hat)

            exposures.append({
                "canary_id":      canary["canary_id"],
                "true_secret":    true_secret,
                "rank_in_sample": rank_in_sample,
                "rank_hat":       rank_hat,
                "exposure":       round(exposure, 4),
            })

            print(f"    {canary['canary_id']}  n_reps={n_reps}  "
                  f"rank={rank_in_sample}  exposure={exposure:.2f} bits")

        mean_exposure = sum(e["exposure"] for e in exposures) / len(exposures) if exposures else 0.0
        results_by_reps[str(n_reps)] = {
            "n":             len(canaries),
            "mean_exposure": round(mean_exposure, 4),
            "per_canary":    exposures,
        }

    return {"by_n_reps": results_by_reps}


# ─── (d) Targeted PII Extraction ─────────────────────────────────────────────

def attack_targeted(victim, victim_tok, targeted_records,
                    max_new_tokens=50, seed=42, n_samples=10):

    random.seed(seed)
    device = next(victim.parameters()).device

    # Split into member / non-member
    member_recs = [r for r in targeted_records if r["split"] == "member"]
    control_recs = [r for r in targeted_records if r["split"] != "member"]

    def run_on_split(records, split_name):
        top1_leaks = 0
        top10_leaks = 0
        total = len(records)

        for i, rec in enumerate(records):
            prompt     = rec["prompt"]
            gold_value = rec["gold_value"]

            # ── Greedy (top-1) ────────────────────────────────────────────
            inputs = victim_tok(prompt, return_tensors="pt",
                                truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = victim.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=victim_tok.eos_token_id,
                )
            new_tokens = out[0][inputs.input_ids.shape[1]:]
            greedy_out = victim_tok.decode(new_tokens, skip_special_tokens=True)

            if gold_value.lower() in greedy_out.lower():
                top1_leaks += 1

            # ── Sampled (top-10) ──────────────────────────────────────────
            found_in_samples = False
            for _ in range(n_samples):
                with torch.no_grad():
                    out_s = victim.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9,
                        pad_token_id=victim_tok.eos_token_id,
                    )
                new_tokens_s = out_s[0][inputs.input_ids.shape[1]:]
                sampled_out  = victim_tok.decode(new_tokens_s, skip_special_tokens=True)
                if gold_value.lower() in sampled_out.lower():
                    found_in_samples = True
                    break   # early exit — found in top-10

            if found_in_samples or (gold_value.lower() in greedy_out.lower()):
                top10_leaks += 1

            if (i + 1) % 50 == 0:
                print(f"    [{split_name}] {i+1}/{total}  "
                      f"top1={top1_leaks}  top10={top10_leaks}", flush=True)

        top1_rate  = top1_leaks  / total if total > 0 else 0.0
        top10_rate = top10_leaks / total if total > 0 else 0.0
        return top1_rate, top10_rate, total

    print(f"  [Targeted] Evaluating {len(member_recs)} member prompts …")
    top1_m, top10_m, n_m = run_on_split(member_recs, "member")

    print(f"  [Targeted] Evaluating {len(control_recs)} non-member prompts …")
    top1_c, top10_c, n_c = run_on_split(control_recs, "non-member")

    return {
        "n_prompts":           len(targeted_records),
        "n_member":            n_m,
        "n_control":           n_c,
        "top1_leak_rate":      round(top1_m,  4),
        "top10_leak_rate":     round(top10_m, 4),
        "control_top1_leak_rate":  round(top1_c,  4),
        "control_top10_leak_rate": round(top10_c, 4),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Extraction Attacks — victim: {args.victim_name}")
    print(f"  Attack mode: {args.attack}")
    print(f"{'='*60}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[1/2] Loading victim model + adapter …")
    victim, victim_tok = load_model(args.victim_model, args.adapter_path)

    print("[2/2] Loading reference model (no fine-tuning) …")
    ref, ref_tok = load_model(args.ref_model, adapter_path=None)

    # ── Load data ─────────────────────────────────────────────────────────────
    corpus_records   = load_jsonl(args.corpus)
    canary_records   = load_jsonl(args.canaries)
    targeted_records = load_jsonl(args.targeted_file)

    # Pre-load corpus PII values for untargeted attack
    load_corpus_pii(corpus_records)

    # ── Held-out perplexity ───────────────────────────────────────────────────
    from train_victim import load_jsonl as _load, compute_perplexity
    heldout_ppl = compute_perplexity(victim, victim_tok,
                                     corpus_records[:100],   # proxy using corpus
                                     max_length=512)
    print(f"\n  Heldout PPL (proxy) = {heldout_ppl:.3f}\n")

    report = {
        "victim":       args.victim_name,
        "corpus":       args.corpus,
        "heldout_ppl":  round(heldout_ppl, 3),
    }

    # ── Run requested attacks ─────────────────────────────────────────────────

    if args.attack in ("discoverable", "all"):
        print("\n" + "─"*50)
        print("Attack (a): Discoverable Memorization")
        print("─"*50)
        result = attack_discoverable(
            victim, victim_tok, corpus_records,
            prefix_len=args.prefix_len,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        report["discoverable"] = result

    if args.attack in ("untargeted", "all"):
        print("\n" + "─"*50)
        print("Attack (b): Untargeted Extraction")
        print("─"*50)
        result = attack_untargeted(
            victim, victim_tok, ref, ref_tok, corpus_records,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        report["untargeted"] = result

    if args.attack in ("canary", "all"):
        print("\n" + "─"*50)
        print("Attack (c): Canary Exposure")
        print("─"*50)
        result = attack_canary_exposure(
            victim, victim_tok, canary_records,
            n_alternatives=args.n_canary_alts,
            seed=args.seed,
        )
        report["canary"] = result

    if args.attack in ("targeted", "all"):
        print("\n" + "─"*50)
        print("Attack (d): Targeted PII Extraction")
        print("─"*50)
        result = attack_targeted(
            victim, victim_tok, targeted_records,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        report["targeted"] = result

    # ── Save report ───────────────────────────────────────────────────────────
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\n✓ Report saved to {out}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  victim: {args.victim_name}")
    if "discoverable" in report:
        print(f"  Discoverable EM rate: {report['discoverable']['exact_match_rate']:.4f}")
    if "untargeted" in report:
        print(f"  Untargeted P@100 (ppl): {report['untargeted']['precision_at_100']['ppl']:.4f}")
    if "canary" in report:
        for reps, v in report["canary"]["by_n_reps"].items():
            print(f"  Canary n_reps={reps}: mean_exposure={v['mean_exposure']:.2f} bits")
    if "targeted" in report:
        t = report["targeted"]
        print(f"  Targeted top-1 (member): {t['top1_leak_rate']:.4f}")
        print(f"  Targeted top-1 (control): {t['control_top1_leak_rate']:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
