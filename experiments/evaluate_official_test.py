"""
用途：在全组方案冻结后，用已训练权重评估官方七类测试子集，供 A 组织独立测试。
核心职责：
- 由 GT-final_test.csv 按文件名固定顺序取出类别 1、2、3、4、5、7、8 的全部测试图。
- 对冻结的比较清单逐模型输出十三条指标、逐图预测、特征与成本记录。
- 记录冻结文件、标注、子集清单、条件像素和源码指纹，保证测试结果可追溯。
关键副作用/安全约束：
- 必须先写出冻结清单；评估时逐项校验权重和配置与冻结时一致，任何改动都会被拒绝。
- 测试标签只用于确定七类子集与计算指标，不进入训练、不用于挑选方法或强度。
- 输出必须写入不存在的新目录，拒绝覆盖。
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from a_conditions import LEVELS, TEST_NOISE_SEED, manifest, test_conditions
from run_c_study import PROJECT, environment_info, file_sha256
from cnn_robustness_study import (
    CLASSES, INPUT_MODES, METRIC_FIELDS, PREDICTION_FIELDS, _evaluate, _metric_row,
    _prediction_rows, _write_csv, tensor_views, verify_completed_run,
)

# 官方测试集在七类限速牌上的固定规模，用作清单校验。
EXPECTED_IMAGES = {1: 720, 2: 750, 3: 450, 4: 660, 5: 630, 7: 450, 8: 450}
TEST_FOLDER = "GTSRB/Final_Test/Images"
IMAGE_SIZE = 32
SOURCE_FILES = ("a_conditions.py", "evaluate_official_test.py")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _load_study(study):
    from run_resnet_study import validate_resnet_config

    config_path = Path(study) / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing study configuration: {config_path}")
    study_config = _read_json(config_path)
    if not isinstance(study_config, dict) or set(study_config) != {"common", "matrix"}:
        raise ValueError(f"Invalid study configuration envelope: {config_path}")
    validate_resnet_config(study_config["common"])
    return study_config, [tuple(row) for row in study_config["matrix"]]


def _validate_run(study, run_id):
    """Integrity-check one frozen run and return its configuration."""
    run = Path(study) / "runs" / run_id
    config = _read_json(run / "config.json")
    verify_completed_run(run, config)
    if config["run_id"] != run_id:
        raise ValueError(f"Run id mismatch: {run}")
    return run, config


def _test_rows(gt_path):
    """Read the official test labels and keep only the seven speed-limit classes."""
    counter = {class_id: 0 for class_id in CLASSES}
    rows = []
    with Path(gt_path).open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle, delimiter=";"):
            class_id = int(record["ClassId"])
            if class_id not in counter:
                continue
            counter[class_id] += 1
            rows.append({
                "image": f"{TEST_FOLDER}/{record['Filename']}",
                "class_id": class_id,
                # The official test split carries no recording-track identifier.
                "track_id": "",
                "split": "test",
            })
    if counter != EXPECTED_IMAGES:
        raise ValueError(f"Official seven-class test subset changed: {counter}")
    rows.sort(key=lambda row: row["image"])
    if len({row["image"] for row in rows}) != sum(EXPECTED_IMAGES.values()):
        raise ValueError("Duplicate official test images in the subset")
    return rows


def _load_images(data_root, rows):
    """Resize with the training-time INTER_AREA rule so the images match the training pipeline."""
    images = []
    for row in rows:
        path = Path(data_root) / row["image"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing official test image: {path}")
        buffer = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode {path}")
        images.append(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA))
    return images


def _subset_manifest_bytes(rows):
    """Deterministic byte form of the frozen subset, quoted so its hash is stable."""
    lines = ["image,class_id"] + [f"{row['image']},{row['class_id']}" for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _init_freeze(studies, destination, gt_path):
    """Snapshot the comparison set before any official test score is produced."""
    if Path(destination).exists():
        raise FileExistsError(f"Freeze file already exists: {destination}")
    names = [Path(study).name for study in studies]
    if len(set(names)) != len(names):
        raise ValueError(f"Study directory names must be unique: {names}")
    entries, fingerprints = [], {}
    for study in studies:
        study_config, matrix = _load_study(study)
        if not matrix:
            raise ValueError(f"Study has no runs: {study}")
        for augmentation, mode, seed in matrix:
            run_id = f"{augmentation}_{mode}_seed{seed}"
            run, config = _validate_run(study, run_id)
            entries.append({
                "study": Path(study).as_posix(), "run_id": run_id,
                "augmentation": augmentation, "input": mode, "seed": seed,
                "config_sha256": file_sha256(run / "config.json"),
                "checkpoint_sha256": file_sha256(run / "model.pt"),
            })
        fingerprints[Path(study).as_posix()] = study_config["common"]["data_fingerprint"]
    if not entries:
        raise ValueError("Freeze needs at least one completed run")
    if len({(entry["study"], entry["run_id"]) for entry in entries}) != len(entries):
        raise ValueError("Flight freeze list contains duplicate runs")
    rows = _test_rows(gt_path)
    freeze = {
        "schema_version": 1,
        "purpose": "Official seven-class test subset evaluated only after the method was frozen",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gt_sha256": file_sha256(gt_path),
        "subset_sha256": _sha256_bytes(_subset_manifest_bytes(rows)),
        "n_test_images": len(rows),
        "conditions": 13,
        "training_data_fingerprints": fingerprints,
        "comparison_set": entries,
        "note": ("Test labels build the seven-class subset and score the runs only. "
                 "No strength, epoch or method may be changed after this file is written."),
    }
    Path(destination).write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    distinct = sorted(set(fingerprints.values()))
    print(f"Wrote freeze for {len(entries)} runs from {len(studies)} studies to {destination}")
    print(f"Frozen subset: {len(rows)} official test images, 13 conditions, "
          f"{len(rows) * 13} image-condition pairs per run")
    if len(distinct) > 1:
        print("WARNING: frozen studies do not share one training pixel fingerprint:")
        for study, value in sorted(fingerprints.items()):
            print(f"  {study}: {value}")
        print("Paired cross-member comparisons then mix pixel pipelines; record this in the report.")
    return freeze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, help="冻结清单；评估必需")
    parser.add_argument("--init-freeze", type=Path, help="先写出冻结清单（只读研究目录）")
    parser.add_argument("--study", type=Path, action="append", default=[],
                        help="配合 --init-freeze 列出研究目录，可重复")
    parser.add_argument("--output", type=Path, help="新的评估输出目录")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/GTSRB/raw")
    parser.add_argument("--gt", type=Path, default=PROJECT / "data/GTSRB/raw/GT-final_test.csv")
    parser.add_argument("--prepare-only", action="store_true", help="只生成并核对测试子集与条件")
    parser.add_argument("--limit", type=int, default=0,
                        help="调试用：只用子集前 N 张图，产物标记为 limited，不进入报告")
    args = parser.parse_args(argv)

    if args.init_freeze:
        if not args.study:
            parser.error("--init-freeze needs at least one --study")
        _init_freeze(args.study, args.init_freeze, args.gt)
        return
    if not args.output:
        parser.error("--output is required")
    if not args.prepare_only and not args.freeze:
        parser.error("Evaluation needs --freeze; --prepare-only may run without it")
    if args.output.exists():
        parser.error(f"Output already exists, choose a new directory: {args.output}")
    if args.limit and args.limit < 2:
        parser.error("--limit must be 0 (all images) or at least 2")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this PyTorch installation")
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    freeze, entries = None, []
    if args.freeze:
        freeze = _read_json(args.freeze)
        if freeze.get("schema_version") != 1:
            raise ValueError("Unsupported freeze file schema")
        entries = freeze["comparison_set"]
        if not entries:
            raise ValueError("Empty comparison set")

    rows = _test_rows(args.gt)
    if freeze is not None:
        if file_sha256(args.gt) != freeze["gt_sha256"]:
            raise ValueError("Official test labels changed after the freeze")
        if _sha256_bytes(_subset_manifest_bytes(rows)) != freeze["subset_sha256"]:
            raise ValueError("Reconstructed test subset differs from the frozen subset")
        if len(rows) != freeze["n_test_images"]:
            raise ValueError("Frozen test subset size mismatch")
    # Fingerprint the full subset before any debug truncation touches `rows`.
    gt_sha256 = file_sha256(args.gt)
    subset_sha256 = _sha256_bytes(_subset_manifest_bytes(rows))

    print(f"Loading {len(rows)} official test images and 13 frozen conditions...", flush=True)
    clean = _load_images(args.data_root, rows)
    specs = list(test_conditions(clean, rows))
    if len(specs) != 13:
        raise ValueError(f"Expected thirteen frozen conditions, got {len(specs)}")
    if args.limit:
        rows = rows[: args.limit]
        specs = [(kind, severity, value, images[: args.limit])
                 for kind, severity, value, images in specs]
    n_images = len(rows)

    args.output.mkdir(parents=True)
    _write_json(args.output / "conditions.json", {
        "schema_version": 1,
        "scope": "official_seven_class_test_subset",
        "levels": LEVELS,
        "noise_seed": TEST_NOISE_SEED,
        "note": ("All twelve interference conditions are generated fresh from the resized test "
                 "images; the three noise strengths share one standard normal field per file."),
        "gt_sha256": gt_sha256,
        "subset_sha256": subset_sha256,
        "freeze_sha256": file_sha256(args.freeze) if args.freeze else None,
        "frozen": freeze is not None,
        "n_images": n_images,
        "source_hashes": {name: file_sha256(Path(__file__).resolve().parent / name)
                          for name in SOURCE_FILES},
        "conditions": manifest(specs),
    })
    _write_csv(args.output / "test_manifest.csv",
               ["image", "class_id", "track_id", "split"], rows)
    if args.limit:
        _write_json(args.output / "debug_limited.json",
                    {"limit": args.limit, "note": "Debug run; exclude from any report or summary."})
    print(f"Frozen {len(specs)} conditions x {n_images} official test images "
          f"({len(specs) * n_images} image-condition pairs per run)", flush=True)
    if args.prepare_only:
        print("Prepared the official test subset and conditions only; no model was loaded.")
        return

    from resnet_robustness_study import ResNet18Fusion

    records, totals = [], {"runs": 0, "metrics": 0, "predictions": 0, "features": 0}
    for entry in entries:
        study, run_id = Path(entry["study"]), entry["run_id"]
        run, config = _validate_run(study, run_id)
        if file_sha256(run / "config.json") != entry["config_sha256"]:
            raise ValueError(f"Run configuration changed after the freeze: {run}")
        if file_sha256(run / "model.pt") != entry["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after the freeze: {run}")
        recorded = _read_json(study / "run_config.json")["common"]["data_fingerprint"]
        if freeze["training_data_fingerprints"].get(study.as_posix()) != recorded:
            raise ValueError(f"Study training pixels changed after the freeze: {study}")
        checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("config") != config:
            raise ValueError(f"Checkpoint configuration mismatch: {run_id}")
        model = ResNet18Fusion(INPUT_MODES[config["input"]][0], pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model = model.to(args.device, memory_format=torch.channels_last).eval()

        label = f"{study.name}/{run_id}"
        destination = args.output / study.name / run_id
        destination.mkdir(parents=True)
        metrics, predictions, costs = [], [], []
        for kind, severity, value, images in specs:
            started = time.perf_counter()
            views = tensor_views(images)
            preprocessing_seconds = time.perf_counter() - started
            spec = {"views": views, "rows": rows, "split": "test",
                    "condition": kind, "severity": severity}
            warmup = min(128, n_images)
            _evaluate(model, config["input"], dict(
                spec, rows=rows[:warmup],
                views={key: tensor[:warmup] for key, tensor in views.items()}),
                config["classes"], 128)
            output, elapsed = _evaluate(model, config["input"], spec, config["classes"], 128)
            metrics.append(_metric_row(config, spec, output,
                                       sum(parameter.numel() for parameter in model.parameters())))
            predictions.extend(_prediction_rows(config, spec, output))
            np.savez_compressed(destination / f"{kind}_{severity}.npz", **output)
            costs.append({
                "condition": kind, "severity": severity, "parameter": "" if value is None else value,
                "images": n_images, "batch_size": 128,
                "preprocessing_seconds": preprocessing_seconds, "evaluation_seconds": elapsed,
            })
            print(f"{label} {kind}/{severity}: "
                  f"macro_f1={metrics[-1]['macro_f1']:.4f} accuracy={metrics[-1]['accuracy']:.4f} "
                  f"inference={elapsed:.3f}s", flush=True)
        _write_csv(destination / "metrics.csv", METRIC_FIELDS, metrics)
        _write_csv(destination / "predictions.csv", PREDICTION_FIELDS, predictions)
        _write_csv(destination / "costs.csv", list(costs[0]), costs)
        if len(metrics) != 13 or len(predictions) != 13 * n_images:
            raise ValueError(f"Unexpected artifact size for {label}")
        records.append({"study": study.as_posix(), "run_id": run_id, "label": label,
                        "metrics_rows": len(metrics), "prediction_rows": len(predictions)})
        totals["runs"] += 1
        totals["metrics"] += len(metrics)
        totals["predictions"] += len(predictions)
        totals["features"] += len(specs)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    completion = dict(
        schema_version=1, scope="official_seven_class_test_subset",
        freeze=str(args.freeze.as_posix()),
        conditions=13, images=n_images, limited=bool(args.limit),
        runs=records, totals=totals,
        expected_totals={"runs": len(entries), "metrics": 13 * len(entries),
                         "predictions": 13 * len(entries) * n_images,
                         "features": 13 * len(entries)},
        environment=environment_info(args.device),
        artifact_hashes={path.relative_to(args.output).as_posix(): file_sha256(path)
                         for path in sorted(args.output.rglob("*")) if path.is_file()},
    )
    if totals != completion["expected_totals"]:
        raise ValueError("Evaluated totals differ from the frozen expectations")
    _write_json(args.output / "completion.json", completion)
    print(f"Completed the official test subset for {totals['runs']} frozen runs: "
          f"{totals['metrics']} metric rows, {totals['predictions']} predictions.")
    print("Report this as an independent test; do not reuse these scores to change the method.")


if __name__ == "__main__":
    main()
