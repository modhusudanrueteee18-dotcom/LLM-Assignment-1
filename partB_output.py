import json
import csv
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Modhusudan\PycharmProjects\LLM\results")
OUTPUT_FILE = DATA_DIR / "partB_valuable_information.csv"


def load_json(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(value):
    """Convert a decimal metric to percentage."""
    if value is None:
        return ""
    return round(value * 100, 3)


def add_row(rows, table, data):
    """Append a row to the final CSV dataset."""
    row = {
        "table": table,
        **data
    }
    rows.append(row)

files = {
    "A5 fine-tuned": "partB_metrics.json",
    "Regex": "partB_metrics_regex.json",
    "Presidio": "partB_metrics_presidio.json",
    "Qwen zero-shot": "partB_metrics_zeroshot.json",
}

ood_file = "partB_metrics_ood.json"

rows = []

for method_name, filename in files.items():

    data = load_json(filename)

    exact = data["exact"]
    iou = data["overlap_iou50"]

    add_row(
        rows,
        "Overall ID Test",
        {
            "system": method_name,
            "test_set": data["test_set"],
            "n": data["n"],

            "precision_exact": pct(exact["micro_p"]),
            "recall_exact": pct(exact["micro_r"]),
            "f1_exact": pct(exact["micro_f1"]),
            "macro_f1_exact": pct(exact["macro_f1"]),

            "f1_iou50": pct(iou["micro_f1"]),
            "macro_f1_iou50": pct(iou["macro_f1"]),

            "leak_rate": pct(data["leak_rate"]),
            "over_masking_rate": pct(data["over_masking_rate"]),
            "well_formed_rate": pct(data["well_formed_rate"]),
        }
    )


ood = load_json(ood_file)

exact = ood["exact"]
iou = ood["overlap_iou50"]

add_row(
    rows,
    "OOD Test",
    {
        "system": "A5 fine-tuned",
        "test_set": ood["test_set"],
        "n": ood["n"],

        "precision_exact": pct(exact["micro_p"]),
        "recall_exact": pct(exact["micro_r"]),
        "f1_exact": pct(exact["micro_f1"]),
        "macro_f1_exact": pct(exact["macro_f1"]),

        "f1_iou50": pct(iou["micro_f1"]),
        "macro_f1_iou50": pct(iou["macro_f1"]),

        "leak_rate": pct(ood["leak_rate"]),
        "over_masking_rate": pct(ood["over_masking_rate"]),
        "well_formed_rate": pct(ood["well_formed_rate"]),
    }
)


a5 = load_json("partB_metrics.json")

for label, metrics in a5["exact"]["per_class"].items():

    iou_metrics = (
        a5.get("overlap_iou50", {})
           .get("per_class", {})
           .get(label, {})
    )

    add_row(
        rows,
        "A5 Per-Class",
        {
            "system": "A5 fine-tuned",
            "test_set": a5["test_set"],
            "n": a5["n"],

            "label": label,
            "support": metrics.get("support", ""),

            "precision_exact": pct(metrics.get("p")),
            "recall_exact": pct(metrics.get("r")),
            "f1_exact": pct(metrics.get("f1")),

            "f1_iou50": pct(iou_metrics.get("f1")),
        }
    )


fieldnames = [
    "table",
    "system",
    "test_set",
    "n",
    "label",
    "support",
    "precision_exact",
    "recall_exact",
    "f1_exact",
    "macro_f1_exact",
    "f1_iou50",
    "macro_f1_iou50",
    "leak_rate",
    "over_masking_rate",
    "well_formed_rate",
]


with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(rows)


print("=" * 60)
print("CSV export completed successfully")
print("=" * 60)
print(f"Output file: {OUTPUT_FILE}")
print(f"Total rows: {len(rows)}")
print()

print("Tables included:")
print("1. Overall ID Test")
print("2. OOD Test")
print("3. A5 Per-Class")