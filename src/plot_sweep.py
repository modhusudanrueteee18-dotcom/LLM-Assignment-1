import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


METHOD_COLORS = {"LoRA": "#2563EB", "QLoRA": "#DC2626"}
MARKER_STYLES = {"LoRA": "o", "QLoRA": "s"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/partA_sweep.csv")
    p.add_argument("--out", default="results/partA_fig.png")
    return p.parse_args()


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def main():
    args = parse_args()
    rows = load_csv(args.csv)

    run_ids       = [r["run_id"]          for r in rows]
    methods       = [r["method"]          for r in rows]
    trainable_pct = [float(r["trainable_pct"])   for r in rows]
    val_f1        = [float(r["val_entity_f1"])    for r in rows]
    peak_mem      = [float(r["peak_gpu_mem_gb"])  for r in rows]
    lora_r        = [int(r["lora_r"])             for r in rows]
    lora_alpha    = [int(r["lora_alpha"])         for r in rows]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "PEFT-Qwen2.5-1.5B-Instruct PII Sanitizer",
        fontsize=11, fontweight="bold", y=1.02,
    )

    # ── Panel 1: F1 vs. Trainable % (log x-axis) ─────────────────────────────
    ax1.set_xscale("log")
    for rid, method, tp, f1, r_val, alpha in zip(
            run_ids, methods, trainable_pct, val_f1, lora_r, lora_alpha):
        color  = METHOD_COLORS.get(method, "grey")
        marker = MARKER_STYLES.get(method, "o")
        ax1.scatter(tp, f1, color=color, marker=marker, s=120, zorder=3,
                    label=method if rid == run_ids[0] or method == "QLoRA" else "")
        ax1.annotate(
            f"{rid}\n(r={r_val}, α={alpha})",
            xy=(tp, f1), xytext=(6, 4), textcoords="offset points",
            fontsize=7.5, color=color,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

    # Efficiency frontier (Pareto-optimal points in F1/trainable_pct)
    frontier_x, frontier_y = [], []
    best_f1 = -1
    for tp, f1 in sorted(zip(trainable_pct, val_f1)):
        if f1 > best_f1:
            frontier_x.append(tp)
            frontier_y.append(f1)
            best_f1 = f1
    ax1.step(frontier_x, frontier_y, color="orange", linestyle="--",
             linewidth=1.2, label="Pareto frontier", where="post", alpha=0.7)

    ax1.set_xlabel("Trainable Parameters (% of total, log scale)", fontsize=10)
    ax1.set_ylabel("Validation Entity F1 (Exact Match)", fontsize=10)
    ax1.set_title("F1 vs. Parameter Budget", fontsize=10)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f%%"))
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8, loc="lower right")

    # ── Panel 2: GPU Memory vs. F1 ───────────────────────────────────────────
    for rid, method, mem, f1 in zip(run_ids, methods, peak_mem, val_f1):
        color  = METHOD_COLORS.get(method, "grey")
        marker = MARKER_STYLES.get(method, "o")
        ax2.scatter(mem, f1, color=color, marker=marker, s=120, zorder=3)
        ax2.annotate(rid, xy=(mem, f1), xytext=(4, 4),
                     textcoords="offset points", fontsize=8.5, color=color)

    # Highlight QLoRA memory saving
    qlora_rows = [(mem, f1) for m, mem, f1 in zip(methods, peak_mem, val_f1) if m == "QLoRA"]
    if qlora_rows:
        qmem, qf1 = qlora_rows[0]
        ax2.annotate(
            "← QLoRA\nmemory saving",
            xy=(qmem, qf1), xytext=(qmem + 0.4, qf1 - 0.02),
            fontsize=7.5, color="#DC2626",
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.8),
        )

    ax2.set_xlabel("Peak GPU Memory (GB)", fontsize=10)
    ax2.set_ylabel("Validation Entity F1 (Exact Match)", fontsize=10)
    ax2.set_title("Quality vs. Memory Footprint", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Shared legend for method colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB",
               markersize=9, label="LoRA"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#DC2626",
               markersize=9, label="QLoRA"),
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved figure → {out}")

    # Also save PDF companion
    pdf_out = out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"Saved figure → {pdf_out}")


if __name__ == "__main__":
    main()
