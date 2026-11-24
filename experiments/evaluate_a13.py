"""
用途：用已核验的训练产物在开发集上执行 A 的冻结十三条件评估与计时。
核心职责：
- 校验研究目录、逐运行配置、数据指纹和完成标记，再加载最终权重推理。
- 对每个运行输出十三条 Macro-F1/accuracy、逐图预测、十三份 512 维特征与耗时表。
- 记录条件参数、像素指纹、划分与源码指纹，形成可复核的评估清单。
关键副作用/安全约束：
- 只读研究目录；输出必须写入不存在的新目录，拒绝覆盖。
- 不训练、不选轮次、不按结果调整强度，也不读取官方测试图像。
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from a_conditions import LEVELS, NOISE_SEED, conditions, manifest
from run_c_study import PROJECT, environment_info, file_sha256, prepare_data
from cnn_robustness_study import (
    INPUT_MODES, METRIC_FIELDS, PREDICTION_FIELDS, _evaluate, _metric_row, _prediction_rows,
    _write_csv, tensor_views, verify_completed_run,
)

SOURCE_FILES = ("a_conditions.py", "evaluate_a13.py")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_study(study):
    """Read and validate the study envelope without touching models or CUDA."""
    from run_resnet_study import validate_resnet_config

    config_path = study / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing study configuration: {config_path}")
    study_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(study_config, dict) or set(study_config) != {"common", "matrix"}:
        raise ValueError(f"Invalid study configuration envelope: {config_path}")
    validate_resnet_config(study_config["common"])
    matrix = [tuple(row) for row in study_config["matrix"]]
    if len({row for row in matrix}) != len(matrix):
        raise ValueError("Study matrix contains duplicate runs")
    return study_config, matrix


def _load_run(study, run_id, data_fingerprint):
    """Validate one completed run and return its frozen config and reloadable model."""
    from resnet_robustness_study import ResNet18Fusion

    run = study / "runs" / run_id
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    verify_completed_run(run, config)
    if config["run_id"] != run_id:
        raise ValueError(f"Run id mismatch: {run}")
    if config["data_fingerprint"] != data_fingerprint:
        raise ValueError(
            f"Run {run_id} was trained on different generated pixels; "
            "rerun the evaluation in the training environment"
        )
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("config") != config:
        raise ValueError(f"Checkpoint configuration mismatch: {run_id}")
    model = ResNet18Fusion(INPUT_MODES[config["input"]][0], pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return run, config, model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, help="已完成核验的 ResNet18 研究目录；评估时必需")
    parser.add_argument("--output", type=Path, required=True, help="新的评估输出目录")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/GTSRB/raw")
    parser.add_argument("--split", type=Path, default=PROJECT / "results/pilot_split.csv")
    parser.add_argument("--prepare-only", action="store_true",
                        help="只生成并核对十三个条件，不加载模型")
    parser.add_argument("--limit", type=int, default=0,
                        help="调试用：只用前 N 张开发集图像，产物标记为 limited，不进入报告")
    parser.add_argument("--expect-runs", type=int, default=0,
                        help="预期运行数（A 的正式九次矩阵填 9），不匹配则拒绝")
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"Output already exists, choose a new directory: {args.output}")
    if args.limit < 0 or 0 < args.limit < 2:
        parser.error("--limit must be 0 (all images) or at least 2 for a meaningful debug run")
    if not args.prepare_only and args.study is None:
        parser.error("--study is required unless --prepare-only is used")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this PyTorch installation")
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    matrix = []
    if args.study is not None:
        _, matrix = _load_study(args.study)
        if args.expect_runs and len(matrix) != args.expect_runs:
            raise ValueError(f"Study has {len(matrix)} runs, expected {args.expect_runs}")

    print(f"Preparing frozen development data for {len(matrix)} runs...", flush=True)
    data = prepare_data(args.data_root, args.split)
    specs = list(conditions(data))
    if len(specs) != 13:
        raise ValueError(f"Expected thirteen frozen conditions, got {len(specs)}")

    rows = data["validation_rows"]
    if args.limit:
        rows = rows[: args.limit]
        specs = [(kind, severity, value, images[: args.limit])
                 for kind, severity, value, images in specs]
    n_images = len(rows)

    args.output.mkdir(parents=True)
    condition_manifest = dict(
        schema_version=1,
        scope="development_validation_only",
        levels=LEVELS,
        noise_seed=NOISE_SEED,
        note=("Medium conditions reuse prepare_data pixels; low/high noise derive a per-file "
              "seed from the frozen image order so both strengths share one normal field."),
        data_fingerprint=data["data_fingerprint"],
        split_sha256=file_sha256(args.split),
        study=args.study.as_posix() if args.study else None,
        study_config_sha256=(file_sha256(args.study / "run_config.json")
                             if args.study else None),
        source_hashes={name: file_sha256(Path(__file__).resolve().parent / name)
                       for name in SOURCE_FILES},
        conditions=manifest(specs),
    )
    _write_json(args.output / "conditions.json", condition_manifest)
    _write_csv(args.output / "images.csv", ["image", "class_id", "track_id", "split"], rows)
    if args.limit:
        _write_json(args.output / "debug_limited.json",
                    {"limit": args.limit, "note": "Debug run; exclude from any report or summary."})
    print(f"Frozen {len(specs)} conditions x {n_images} development images ...", flush=True)
    if args.prepare_only:
        print("Prepared conditions and manifest only; no model was loaded.")
        return

    records, totals = [], {"runs": 0, "metrics": 0, "predictions": 0, "features": 0}
    for augmentation, mode, seed in matrix:
        run_id = f"{augmentation}_{mode}_seed{seed}"
        run, config, model = _load_run(args.study, run_id, data["data_fingerprint"])
        model = model.to(args.device, memory_format=torch.channels_last).eval()
        destination = args.output / run_id
        destination.mkdir()
        metrics, predictions, costs = [], [], []
        for kind, severity, value, images in specs:
            started = time.perf_counter()
            views = tensor_views(images)
            preprocessing_seconds = time.perf_counter() - started
            spec = {"views": views, "rows": rows, "split": "validation",
                    "condition": kind, "severity": severity}
            warmup = min(128, n_images)
            # Warm up through the identical path: device transfer, resize/crop, forward, copy back.
            _evaluate(model, mode, dict(spec, rows=rows[:warmup],
                                        views={key: value_[:warmup] for key, value_ in views.items()}),
                      config["classes"], 128)
            output, elapsed = _evaluate(model, mode, spec, config["classes"], 128)
            metrics.append(_metric_row(config, spec, output,
                                       sum(parameter.numel() for parameter in model.parameters())))
            predictions.extend(_prediction_rows(config, spec, output))
            np.savez_compressed(destination / f"{kind}_{severity}.npz", **output)
            costs.append({
                "condition": kind, "severity": severity, "parameter": "" if value is None else value,
                "images": n_images, "batch_size": 128,
                "preprocessing_seconds": preprocessing_seconds, "evaluation_seconds": elapsed,
            })
            print(f"{run_id} {kind}/{severity}: "
                  f"macro_f1={metrics[-1]['macro_f1']:.4f} accuracy={metrics[-1]['accuracy']:.4f} "
                  f"inference={elapsed:.3f}s", flush=True)
        _write_csv(destination / "metrics.csv", METRIC_FIELDS, metrics)
        _write_csv(destination / "predictions.csv", PREDICTION_FIELDS, predictions)
        _write_csv(destination / "costs.csv", list(costs[0]), costs)
        expected = len(specs) * n_images
        if len(metrics) != len(specs) or len(predictions) != expected:
            raise ValueError(f"Unexpected artifact size for {run_id}")
        records.append({
            "run_id": run_id, "augmentation": augmentation, "input": mode, "seed": seed,
            "checkpoint_sha256": file_sha256(run / "model.pt"),
            "metrics_rows": len(metrics), "prediction_rows": len(predictions),
            "feature_files": len(specs),
        })
        totals["runs"] += 1
        totals["metrics"] += len(metrics)
        totals["predictions"] += len(predictions)
        totals["features"] += len(specs)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    completion = dict(
        schema_version=1, scope="development_validation_only",
        study=args.study.as_posix(), conditions=13, images=n_images,
        runs=records, totals=totals,
        expected_totals={"runs": len(matrix), "metrics": len(matrix) * 13,
                         "predictions": len(matrix) * 13 * n_images, "features": len(matrix) * 13},
        limited=bool(args.limit),
        environment=environment_info(args.device),
        artifact_hashes={path.relative_to(args.output).as_posix(): file_sha256(path)
                         for path in sorted(args.output.rglob("*")) if path.is_file()},
    )
    if totals != completion["expected_totals"]:
        raise ValueError("Evaluated totals differ from the frozen expectations")
    _write_json(args.output / "completion.json", completion)
    print(f"Completed {totals['runs']} runs, {totals['metrics']} metric rows, "
          f"{totals['predictions']} predictions, {totals['features']} feature files in {args.output}")
    print("This is the development validation set only; the official test set stays untouched.")


if __name__ == "__main__":
    main()
