"""
用途：按冻结协议运行或核验 C 部分的七类交通标志融合实验。
核心职责：
- 复用既有划分和干扰随机序列，记录数据与环境指纹。
- 调用公共训练程序，运行融合矩阵并支持经校验的恢复。
关键副作用/安全约束：
- 仅写指定的新结果目录，不读取官方测试图像，不覆盖已有运行。
"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np
import torch

import cnn_robustness_pilot as pilot

PROJECT = Path(__file__).resolve().parents[1]
MODES = ("raw", "clahe", "denoised_clahe", "dual", "fixed", "dual_denoised", "dual_spatial")
AUGMENTATIONS = ("repeat_clean", "targeted_motion", "mixed")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(data_root, split_path):
    """Replay the pilot's RNG stream, preserving row order and all corruptions."""
    started = time.perf_counter()
    data_root, split_path = Path(data_root), Path(split_path)
    rng = np.random.default_rng(pilot.SPLIT_SEED)
    expected = pilot.select_split(data_root, rng)
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = [(row["split"], row["image"], int(row["class_id"]), row["track_id"]) for row in rows]
    if actual != expected:
        raise ValueError("Fixed split differs from the original pilot protocol")
    for row in rows:
        row["class_id"] = int(row["class_id"])
    train_rows = [row for row in rows if row["split"] == "train"]
    valid_rows = [row for row in rows if row["split"] == "validation"]
    train_tracks = {row["track_id"] for row in train_rows}
    valid_tracks = {row["track_id"] for row in valid_rows}
    if (len(train_rows), len(valid_rows), len(train_tracks), len(valid_tracks)) != (2940, 840, 98, 28):
        raise ValueError("Unexpected split sizes")
    if train_tracks & valid_tracks or len({row["image"] for row in rows}) != 3780:
        raise ValueError("Duplicate image or overlapping tracks")
    extras_rng = copy.deepcopy(rng)
    extra_indices = extras_rng.choice(len(train_rows), len(train_rows) // 2, replace=False)
    training, targets, validation, valid_targets = pilot.prepare_images(cv2, data_root, expected, rng)
    clean = training["mixed"][:2940]
    training["repeat_clean"] = clean + [clean[index] for index in extra_indices]
    if any(len(images) != 4410 for images in training.values()) or any(
            len(images) != 840 for images in validation.values()):
        raise ValueError("Unexpected augmentation or evaluation budget")
    # Hash source bytes as well as generated pixels; metadata paths stay relative.
    source_files = [{"image": row["image"], "sha256": file_sha256(data_root / row["image"])}
                    for row in rows]
    digest = hashlib.sha256()
    digest.update(json.dumps(source_files, sort_keys=True).encode())
    digest.update(json.dumps(actual, separators=(",", ":")).encode())
    generated = {}
    for category, sets in (("training", training), ("validation", validation)):
        for name, images in sorted(sets.items()):
            pixels = np.ascontiguousarray(images, dtype=np.uint8)
            pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
            generated[f"{category}/{name}"] = pixel_hash
            digest.update(f"{category}/{name}:{pixel_hash}".encode())
    return {"training": training, "validation": validation, "train_targets": targets,
            "validation_targets": valid_targets, "train_rows": train_rows,
            "validation_rows": valid_rows, "extra_indices": extra_indices.tolist(),
            "augmentation_counts": {"repeat_clean": {"repeat": 1470},
                                    "targeted_motion": {"motion": 1470},
                                    "mixed": {"motion": 490, "noise": 490, "dark": 490}},
            "data_fingerprint": digest.hexdigest(), "source_files": source_files,
            "generated_fingerprints": generated,
            "preparation_seconds": time.perf_counter() - started}


def run_matrix(inputs=None, augmentations=None, seeds=(42, 43, 44)):
    if inputs is None and augmentations is None:
        pairs = [(augmentation, "dual") for augmentation in AUGMENTATIONS]
        pairs += [("mixed", "fixed"), ("mixed", "dual_denoised")]
    else:
        pairs = [(augmentation, mode) for augmentation in (augmentations or AUGMENTATIONS)
                 for mode in (inputs or ("dual",))]
    return [(augmentation, mode, seed) for augmentation, mode in pairs for seed in seeds]


def environment_info(device):
    import sklearn
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "opencv": cv2.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "device": device,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/GTSRB/raw")
    parser.add_argument("--split", type=Path, default=PROJECT / "results/pilot_split.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/study_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--inputs", nargs="+", choices=MODES)
    parser.add_argument("--augmentations", nargs="+", choices=AUGMENTATIONS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs/threads must be positive and seeds unique")
    for values in (args.inputs, args.augmentations):
        if values and len(set(values)) != len(values):
            parser.error("duplicate inputs or augmentations")
    from cnn_robustness_study import fit_run, verify_completed_run
    matrix = run_matrix(args.inputs, args.augmentations, args.seeds)
    if args.verify_only:
        for augmentation, mode, seed in matrix:
            run_dir = args.output_dir / "runs" / f"{augmentation}_{mode}_seed{seed}"
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            if (config["augmentation"], config["input"], config["seed"], config["epochs"]) != (
                    augmentation, mode, seed, args.epochs):
                raise ValueError(f"Run does not match requested matrix: {run_dir.name}")
            verify_completed_run(run_dir, config)
        print(f"Verified {len(matrix)} completed runs", flush=True)
        return
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output exists; choose a new directory or --resume: {args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this PyTorch installation")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print("Preparing frozen data and checking fingerprints...", flush=True)
    data = prepare_data(args.data_root, args.split)
    env = environment_info(device)
    code = {name: file_sha256(Path(__file__).parent / name) for name in
            ("run_c_study.py", "cnn_robustness_study.py", "cnn_robustness_pilot.py")}
    # Timings are measured once per invocation and do not change the resume identity.
    common = {"schema_version": 1, "classes": list(pilot.CLASSES), "epochs": args.epochs,
              "batch_size": 128, "learning_rate": 0.003, "optimizer": "AdamW",
              "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
              "image_size": 32, "threads": args.threads, "device": device,
              "training_entries": 4410, "clean_training_images": 2940,
              "validation_images": 840, "train_tracks": 98, "validation_tracks": 28,
              "data_fingerprint": data["data_fingerprint"],
              "split_sha256": file_sha256(args.split), "code_fingerprints": code,
              "environment": env, "evaluation": "final_epoch_development_validation",
              "clahe": {"clip_limit": 2.0, "tile_grid": [4, 4], "channel": "LAB_L"},
              "denoising": {"method": "bilateral", "d": 5, "sigmaColor": 25, "sigmaSpace": 2}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_config.json"
    study_config = {"common": common, "matrix": [list(row) for row in matrix]}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != study_config:
            raise ValueError("Existing study configuration differs; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(study_config, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.copyfile(args.split, args.output_dir / "split.csv")
        manifest = {key: data[key] for key in ("source_files", "generated_fingerprints",
                    "extra_indices", "augmentation_counts", "data_fingerprint", "preparation_seconds")}
        (args.output_dir / "data_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    from cnn_robustness_study import tensor_views
    evaluations, preprocess = {}, {}
    for name, images, rows, split, condition in [
        ("train_clean", data["training"]["mixed"][:2940], data["train_rows"], "train", "clean"),
        *[(f"validation_{condition}", images, data["validation_rows"], "validation", condition)
          for condition, images in data["validation"].items()]]:
        started = time.perf_counter()
        evaluations[name] = {"views": tensor_views(images), "rows": rows,
                             "split": split, "condition": condition}
        preprocess[name] = time.perf_counter() - started
    train_targets = torch.tensor(data["train_targets"], dtype=torch.long)
    for augmentation in dict.fromkeys(row[0] for row in matrix):
        needed = [row for row in matrix if row[0] == augmentation]
        train_views = None
        for _, mode, seed in needed:
            run_id = f"{augmentation}_{mode}_seed{seed}"
            run_dir = args.output_dir / "runs" / run_id
            config = dict(common, run_id=run_id, seed=seed, input=mode, augmentation=augmentation)
            if run_dir.exists():
                saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                comparable = {key: value for key, value in saved.items() if key != "preprocessing_seconds"}
                if comparable != config:
                    raise ValueError(f"Existing run configuration differs: {run_id}")
                verify_completed_run(run_dir, saved)
                print(f"Verified and skipped {run_id}", flush=True)
                continue
            if train_views is None:
                started = time.perf_counter()
                train_views = tensor_views(data["training"][augmentation])
                preprocess[f"training_{augmentation}"] = time.perf_counter() - started
            config["preprocessing_seconds"] = dict(preprocess, data_preparation=data["preparation_seconds"])
            print(f"Running {run_id} ({args.epochs} epochs, {device})", flush=True)
            fit_run(run_dir, config, train_views, train_targets, evaluations)
            verify_completed_run(run_dir, config)
        del train_views
    print(f"Completed and verified {len(matrix)} runs in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
