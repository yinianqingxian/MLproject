#	复算 Macro-F1，计算门控与固定融合的逐种子差值、均值和标准差。
"""Run from any working directory with python path/to/d_analysis/run_analysis.py."""

import csv
from pathlib import Path

from paired_metrics import compare_models
from prediction_data import load_predictions


def main():
    results = Path(__file__).resolve().parents[1] / "results"
    labels, groups = load_predictions(results)
    rows = compare_models(labels, groups)

    output = results / "d_analysis" / "c_gated_minus_fixed.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['condition']:8} {row['mean_pp']:+.2f} pp")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
