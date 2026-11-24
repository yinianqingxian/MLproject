"""
用途：导出 A 负责的数据协议、训练预算与评估预算审计，供 D 复核轨迹交叉和样本数。
核心职责：
- 复用 C 的冻结 `prepare_data`，重放划分、增强索引和干扰随机序列。
- 逐条记录训练样本来源，输出预算表、九次运行矩阵和十三条件评估预算。
- 记录数据指纹、划分指纹和生成环境，明确交叉环境不能假设像素相同。
关键副作用/安全约束：
- 只读原始数据与冻结划分，只写指定输出目录；不训练、不读官方测试图像。
"""

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))

from a_conditions import conditions  # noqa: E402
from run_c_study import file_sha256, prepare_data, run_matrix  # noqa: E402


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(data, split_path):
    """Recompute and assert every budget the protocol promises before writing anything."""
    checks = {}
    train_rows, validation_rows = data["train_rows"], data["validation_rows"]
    train_tracks = {row["track_id"] for row in train_rows}
    validation_tracks = {row["track_id"] for row in validation_rows}
    checks["split_sizes"] = (len(train_rows), len(validation_rows))
    checks["track_counts"] = (len(train_tracks), len(validation_tracks))
    checks["overlapping_tracks"] = sorted(train_tracks & validation_tracks)
    if checks["split_sizes"] != (2940, 840) or checks["track_counts"] != (98, 28):
        raise ValueError(f"Unexpected split: {checks}")
    if checks["overlapping_tracks"]:
        raise ValueError(f"Training and validation tracks overlap: {checks['overlapping_tracks']}")
    if len({row["image"] for row in data["train_rows"] + data["validation_rows"]}) != 3780:
        raise ValueError("Duplicate images in the frozen split")
    if len(set(data["extra_indices"])) != 1470 or len(data["extra_indices"]) != 1470:
        raise ValueError("Extra augmentation indices must be 1470 unique entries")
    if data["augmentation_counts"]["mixed"] != {"motion": 490, "noise": 490, "dark": 490}:
        raise ValueError(f"Unexpected mixed augmentation split: {data['augmentation_counts']}")
    for strategy, images in data["training"].items():
        if len(images) != 4410:
            raise ValueError(f"{strategy} must keep 2940 clean + 1470 augmented entries")
    for condition, images in data["validation"].items():
        if len(images) != 840:
            raise ValueError(f"Validation condition {condition} must have 840 images")
    checks["training_entries"] = {name: len(images) for name, images in data["training"].items()}
    checks["validation_entries"] = {name: len(images) for name, images in data["validation"].items()}
    checks["class_order"] = [1, 2, 3, 4, 5, 7, 8]
    checks["split_sha256"] = file_sha256(split_path)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/GTSRB/raw")
    parser.add_argument("--split", type=Path, default=PROJECT / "results/pilot_split.csv")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/a_data_audit")
    args = parser.parse_args(argv)

    print("Replaying the frozen split, augmentations and corruptions...", flush=True)
    data = prepare_data(args.data_root, args.split)
    checks = audit(data, args.split)
    args.output.mkdir(parents=True, exist_ok=True)

    budget, sources = [], []
    for strategy, counts in data["augmentation_counts"].items():
        budget.append({
            "strategy": strategy, "clean_entries": 2940, "extra_entries": sum(counts.values()),
            "total_entries": len(data["training"][strategy]), "extra_breakdown": json.dumps(
                counts, sort_keys=True), "batch_size": 128, "epochs": 10,
            "optimizer_updates": 350,
        })
        indices = list(range(2940)) + data["extra_indices"]
        for position, source in enumerate(indices):
            row = data["train_rows"][source]
            sources.append({
                "strategy": strategy, "entry": position, "source_index": source,
                "is_extra": position >= 2940, **row,
            })

    evaluation_budget = []
    for kind, severity, value, images in conditions(data):
        evaluation_budget.append({
            "condition": kind, "severity": severity,
            "parameter": "" if value is None else value, "images": len(images),
            "image_condition_pairs": len(images),
        })

    write_csv(args.output / "budget.csv", budget)
    write_csv(args.output / "evaluation_budget.csv", evaluation_budget)
    write_csv(args.output / "a_run_matrix.csv", [
        {"run_id": f"{augmentation}_{mode}_seed{seed}", "augmentation": augmentation,
         "input": mode, "seed": seed}
        for augmentation, mode, seed in run_matrix(inputs=["raw"])
    ])
    write_csv(args.output / "training_sources.csv", sources)
    write_csv(args.output / "validation_manifest.csv",
              [{"condition": condition, **row} for condition in data["validation"]
               for row in data["validation_rows"]])

    import cv2
    import numpy

    summary = {
        "schema_version": 1,
        "status": "data_protocol_passed; no training and no official test evaluation",
        "checks": checks,
        "data_fingerprint": data["data_fingerprint"],
        "generated_fingerprints": data["generated_fingerprints"],
        "extra_indices": data["extra_indices"],
        "augmentation_counts": data["augmentation_counts"],
        "evaluation_budget": {"conditions": len(evaluation_budget), "images": 840,
                              "image_condition_pairs": 840 * len(evaluation_budget),
                              "total_conditions_per_run": len(evaluation_budget)},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "opencv": cv2.__version__, "numpy": numpy.__version__,
            "note": ("The data fingerprint hashes generated pixels and interpolation results, so it "
                     "is only comparable inside one dependency environment. Compare fingerprints "
                     "across machines before any paired comparison."),
        },
        "source_files": data["source_files"],
        "preparation_seconds": data["preparation_seconds"],
    }
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "split_sizes": checks["split_sizes"], "track_counts": checks["track_counts"],
        "overlapping_tracks": checks["overlapping_tracks"],
        "training_entries": checks["training_entries"],
        "evaluation_pairs": summary["evaluation_budget"]["image_condition_pairs"],
        "data_fingerprint": data["data_fingerprint"],
        "opencv": cv2.__version__,
    }, indent=2, ensure_ascii=False))
    print(f"Wrote the audit to {args.output}")


if __name__ == "__main__":
    main()
