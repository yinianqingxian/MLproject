"""
用途：对训练完成的空间门控模型做无需训练的干预对照（正常 7x7 权重图 vs 每图平均后广播）。
核心职责：
- 同一份权重、同一批输入，只切换权重图是否随位置变化，输出两套指标与逐图预测。
- 记录每张图权重图的空间标准差，作为“权重是否真的随位置变化”的直接证据。
- 记录推理耗时。
关键副作用/安全约束：
- 只读模型目录，写新的输出目录；不训练、不改权重、不写回任何训练目录。
"""

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from a_conditions import conditions, pixel_hash  # noqa: E402
from run_c_study import PROJECT, environment_info, file_sha256, prepare_data  # noqa: E402
from cnn_robustness_study import _write_csv, tensor_views, verify_completed_run  # noqa: E402

METRIC_FIELDS = ("run_id", "seed", "mode", "condition", "severity",
                 "macro_f1", "accuracy", "n_images", "gate_mean", "gate_sd")
PREDICTION_FIELDS = ("run_id", "seed", "mode", "image", "track_id", "condition", "severity",
                     "y_true", "y_pred", "gate", "gate_sd")


def macro_f1(y_true, y_pred, classes):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.output.exists():
        raise SystemExit(f"输出目录已存在：{args.output}")
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from resnet_robustness_study import ResNet18Fusion

    data = prepare_data(PROJECT / "data/GTSRB/raw", PROJECT / "results/pilot_split.csv")
    specs = list(conditions(data))
    rows = data["validation_rows"]
    order = sorted(range(len(rows)), key=lambda index: rows[index]["image"])
    names = [rows[index]["image"] for index in order]
    tracks = [rows[index].get("track_id", "") for index in order]
    y_true = np.asarray([rows[index]["class_id"] for index in order], dtype=np.int64)
    classes = [1, 2, 3, 4, 5, 7, 8]

    args.output.mkdir(parents=True)
    manifest = {
        "study": str(args.study), "run_ids": args.run_ids,
        "data_fingerprint": data["data_fingerprint"],
        "condition_pixels": {f"{k}/{s}": pixel_hash(im) for k, s, _, im in specs},
        "environment": environment_info(args.device),
        "modes": ["normal", "average"],
        "note": "average = 每张图 49 个权重取平均后填回全部位置；权重与输入完全相同",
    }

    metrics, predictions, timings = [], [], {}
    for run_id in args.run_ids:
        run_dir = args.study / "runs" / run_id
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        verify_completed_run(run_dir, config)
        if config["input"] != "dual_spatial":
            raise SystemExit(f"{run_id} 不是 dual_spatial 运行")
        checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
        model = ResNet18Fusion("spatial", pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model = model.to(args.device, memory_format=torch.channels_last).eval()

        for condition, severity, _, images in specs:
            views = tensor_views(images)
            raw_tensor = views["raw"].to(args.device, memory_format=torch.channels_last)
            enhanced = views["denoised_clahe"].to(args.device, memory_format=torch.channels_last)
            collected = {mode: {"logits": [], "gate": [], "sd": []} for mode in ("normal", "average")}
            started = time.perf_counter()
            with torch.inference_mode():
                for start in range(0, len(order), 128):
                    batch = torch.tensor(order[start:start + 128], dtype=torch.long)
                    for mode, average in (("normal", False), ("average", True)):
                        out = model.forward_details(raw_tensor[batch], enhanced[batch],
                                                    spatial_average=average)
                        gate_map = out["gate_map"]
                        collected[mode]["logits"].append(out["logits"].cpu())
                        collected[mode]["gate"].append(out["gate"].reshape(-1).cpu())
                        collected[mode]["sd"].append(
                            gate_map.std(dim=(2, 3)).reshape(-1).cpu())
            elapsed = time.perf_counter() - started
            timings[f"{run_id}|{condition}/{severity}"] = elapsed

            for mode in ("normal", "average"):
                logits = torch.cat(collected[mode]["logits"], dim=0)
                gates = torch.cat(collected[mode]["gate"], dim=0).numpy()
                sds = torch.cat(collected[mode]["sd"], dim=0).numpy()
                y_pred = np.asarray(classes)[logits.argmax(1).numpy()]
                scores = macro_f1(y_true, y_pred, classes)
                metrics.append({
                    "run_id": run_id, "seed": config["seed"], "mode": mode,
                    "condition": condition, "severity": severity,
                    "macro_f1": scores["macro_f1"], "accuracy": scores["accuracy"],
                    "n_images": len(y_true),
                    "gate_mean": float(gates.mean()), "gate_sd": float(sds.mean()),
                })
                for index in range(len(y_true)):
                    predictions.append({
                        "run_id": run_id, "seed": config["seed"], "mode": mode,
                        "image": names[index], "track_id": tracks[index],
                        "condition": condition, "severity": severity,
                        "y_true": int(y_true[index]), "y_pred": int(y_pred[index]),
                        "gate": float(gates[index]), "gate_sd": float(sds[index]),
                    })
            print(f"{run_id} {condition}/{severity}: "
                  f"normal {metrics[-2]['macro_f1']:.4f} average {metrics[-1]['macro_f1']:.4f} "
                  f"| 权重图 SD {metrics[-2]['gate_sd']:.4f} | {elapsed:.1f}s", flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    _write_csv(args.output / "metrics.csv", METRIC_FIELDS, metrics)
    _write_csv(args.output / "predictions.csv", PREDICTION_FIELDS, predictions)
    manifest["timings_seconds"] = timings
    manifest["study_config_sha256"] = file_sha256(args.study / "run_config.json")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n完成：{args.output}（{len(metrics)} 行指标，{len(predictions)} 条预测）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
