import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_report", required=True)
    p.add_argument("--san_report", required=True)
    p.add_argument("--out",        default="results/partC_exposure.png")
    return p.parse_args()


def extract_exposure(report_path):
    with open(report_path) as fh:
        data = json.load(fh)
    by_reps = data.get("canary", {}).get("by_n_reps", {})
    reps_list = []
    mean_exp_list = []
    for n_reps_str, v in sorted(by_reps.items(), key=lambda x: int(x[0])):
        reps_list.append(int(n_reps_str))
        mean_exp_list.append(v["mean_exposure"])
    return reps_list, mean_exp_list


def main():
    args = parse_args()
    reps_raw, exp_raw = extract_exposure(args.raw_report)
    reps_san, exp_san = extract_exposure(args.san_report)

    # Estimator ceiling: log2(10000) ≈ 13.29 bits
    ceiling = math.log2(10000)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(reps_raw, exp_raw, "o-", color="#2563EB", linewidth=2.0,
            markersize=7, label="V-raw (trained on raw PII corpus)")
    ax.plot(reps_san, exp_san, "s--", color="#16A34A", linewidth=2.0,
            markersize=7, label="V-san (trained on sanitized corpus)")

    # Estimator ceiling annotation
    ax.axhline(ceiling, color="red", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(reps_raw[-1] * 0.95, ceiling + 0.2,
            f"Estimator ceiling ≈ {ceiling:.1f} bits\n(rank_sample=1 of 10 000)",
            fontsize=7.5, color="red", ha="right")

    # Random guessing baseline (exposure ≈ 0)
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(reps_raw[0], 0.3, "Random guess (exposure ≈ 0 bits)",
            fontsize=7.5, color="grey")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Repetitions in Training Corpus (log scale)", fontsize=10)
    ax.set_ylabel("Mean Canary Exposure (bits)", fontsize=10)
    ax.set_title(
        "Canary Exposure vs. Repetitions\n"
        "Secret Sharer attack on Qwen2.5-0.5B victim models",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks(sorted(set(reps_raw + reps_san)))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(bottom=-1.0)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Per-canary scatter (individual exposures)
    with open(args.raw_report) as fh:
        raw_data = json.load(fh)
    with open(args.san_report) as fh:
        san_data = json.load(fh)

    for dataset_label, data, color in [("V-raw", raw_data, "#93C5FD"),
                                        ("V-san", san_data, "#86EFAC")]:
        by_reps = data.get("canary", {}).get("by_n_reps", {})
        for n_reps_str, v in by_reps.items():
            for canary_exp in v.get("per_canary", []):
                ax.scatter(int(n_reps_str), canary_exp["exposure"],
                           color=color, s=25, alpha=0.6, zorder=2)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
