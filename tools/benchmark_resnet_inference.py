"""
用途：在同一环境测量单视图与双视图 ResNet18 的评估开销。
核心职责：
- 核验训练产物和冻结开发集，使用相同图片、批量和预热交替计时。
- 分别记录逐次吞吐、配对差值及共享视图建缓存耗时。
关键副作用/安全约束：
- 仅读取已有模型和开发集，写入新目录，不训练或覆盖原结果。
"""

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
MODES = ("raw", "clahe", "dual")
BATCH_SIZE = 128
THREADS = 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-run", type=Path, required=True)
    parser.add_argument("--raw-run", type=Path, help="Optional completed, independently trained raw model.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose an unused output directory")

    # Set the numerical controls before importing PyTorch or initializing CUDA.
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(PROJECT / "experiments"))
    import cv2
    import torch
    from cnn_robustness_study import _evaluate, _select_views, tensor_views, verify_completed_run
    from resnet_robustness_study import ResNet18Fusion
    from run_c_study import environment_info, file_sha256, prepare_data
    from run_resnet_study import validate_resnet_config

    if not torch.cuda.is_available():
        parser.error("A CUDA device is required for the same-device benchmark")
    device = torch.device("cuda:0")
    torch.set_num_threads(THREADS)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    split = PROJECT / "results/pilot_split.csv"
    data = prepare_data(PROJECT / "data/GTSRB/raw", split)
    split_sha = file_sha256(split)

    def load_state(directory, mode):
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        validate_resnet_config(config)
        verify_completed_run(directory, config)
        expected = (mode, "mixed", data["data_fingerprint"], split_sha, 10, BATCH_SIZE)
        actual = tuple(config[k] for k in ("input", "augmentation", "data_fingerprint", "split_sha256", "epochs", "batch_size"))
        if actual != expected:
            raise ValueError(f"Unexpected mode, data, or training budget: {directory.name}")
        checkpoint_path = directory / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError(f"Checkpoint config differs: {directory.name}")
        return config, checkpoint["model_state"], file_sha256(checkpoint_path)

    config, state, fusion_sha = load_state(args.fusion_run, "dual")
    raw_sha = fusion_sha
    single_state = {name: value for name, value in state.items() if not name.startswith("gate.")}
    raw_state = single_state
    if args.raw_run:
        raw_config, raw_state, raw_sha = load_state(args.raw_run, "raw")
        if raw_config["seed"] != config["seed"]:
            raise ValueError("Match the single-branch training seed to the fusion checkpoint")

    def init_model(kind, checkpoint, pooling):
        model = ResNet18Fusion(kind, pretrained=False, pooling=pooling)
        model.load_state_dict(checkpoint, strict=True)
        return model.to(device=device, memory_format=torch.channels_last).eval()

    pooling = config.get("pooling", "gap")
    fusion = init_model("learned", state, pooling)
    single = init_model("single", single_state, pooling)
    raw = init_model("single", raw_state, raw_config.get("pooling", "gap")) if args.raw_run else single
    models = {"raw": raw, "clahe": single, "dual": fusion}
    classes = config["classes"]
    rows = data["validation_rows"]
    timing = []
    cache_seconds = {}
    for condition, images in data["validation"].items():
        started = time.perf_counter()
        views = tensor_views(images)
        cache_seconds[condition] = time.perf_counter() - started
        spec = {"views": views, "rows": rows}
        indices = torch.arange(BATCH_SIZE)
        with torch.inference_mode():
            raw_view, clahe_view = _select_views(views, "dual", indices, device)
            for mode in MODES:
                models[mode](*_select_views(views, mode, indices, device))
            # Removing the unused branch must preserve the same-view logits.
            for view, gate in ((raw_view[:8], 0.0), (clahe_view[:8], 1.0)):
                torch.testing.assert_close(single(view), fusion(raw_view[:8], clahe_view[:8], gate_override=gate),
                                           rtol=1e-4, atol=5e-5)
        for repeat in range(len(MODES)):
            # Rotate execution order so each mode occupies every position once.
            for order, mode in enumerate(MODES[repeat:] + MODES[:repeat], start=1):
                _, seconds = _evaluate(models[mode], mode, spec, classes, BATCH_SIZE)
                timing.append({"condition": condition, "repeat": repeat + 1, "order": order, "input": mode,
                               "images": len(rows), "seconds": seconds, "ms_per_image": 1000 * seconds / len(rows)})
        print(f"Timed {condition}: {len(rows)} images × 3 modes × 3 repeats", flush=True)

    summaries = []
    for condition in cache_seconds:
        by_mode = {mode: [r["ms_per_image"] for r in timing if r["condition"] == condition and r["input"] == mode]
                   for mode in MODES}
        for baseline in ("raw", "clahe"):
            comparison = list(zip(by_mode[baseline], by_mode["dual"], strict=True))
            deltas = [dual - single_time for single_time, dual in comparison]
            summaries.append({"condition": condition, "single_input": baseline,
                              "single_mean_ms_per_image": mean(by_mode[baseline]),
                              "dual_mean_ms_per_image": mean(by_mode["dual"]),
                              "dual_over_single_mean": mean(dual / single_time for single_time, dual in comparison),
                              "paired_delta_mean_ms_per_image": mean(deltas), "paired_delta_sd": stdev(deltas)})

    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in (("timings.csv", timing), ("summary.csv", summaries)):
        with (args.output / name).open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    manifest = {
        "scope": "development_validation_inference_timing_only",
        "fusion_run_id": config["run_id"], "fusion_checkpoint_sha256": fusion_sha,
        "raw_run_id": raw_config["run_id"] if args.raw_run else None, "raw_checkpoint_sha256": raw_sha,
        "raw_source": "independently_trained" if args.raw_run else "shared_fusion_encoder_and_classifier",
        "clahe_source": "shared_fusion_encoder_and_classifier",
        "split_sha256": split_sha, "data_fingerprint": data["data_fingerprint"],
        "source_sha256": file_sha256(Path(__file__)), "environment": environment_info("cuda"),
        "batch_size": BATCH_SIZE, "threads": THREADS, "repeats": 3, "warmup_images_per_mode_condition": BATCH_SIZE,
        "shared_three_view_cache_seconds": cache_seconds,
        "timed_components": "CPU cache selection, device copy, official preprocessing, forward, CPU result collection; CUDA synchronized",
        "excluded_components": "disk IO, corruption generation, shared view cache creation, model loading and output serialization",
        "precision": "FP32 with TF32 disabled",
    }
    (args.output / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(timing)} timings and {len(summaries)} comparisons to {args.output}", flush=True)


if __name__ == "__main__":
    main()
