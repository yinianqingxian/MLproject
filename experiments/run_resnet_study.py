"""
用途：使用官方预训练 ResNet18 按冻结协议运行和核验交通标志融合研究。
核心职责：
- 复用原始划分、32 像素视图、增强矩阵及训练预算。
- 记录官方权重、代码、数据和运行环境的完整来源。
- 只读核验完成产物，严格检查恢复配置并拒绝覆盖不完整运行。
关键副作用/安全约束：
- 仅写指定的新结果目录，不读取官方测试图像，不隐式下载权重。
- 核验模式不创建产物，不要求训练 GPU 或本地权重文件。
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import time

import cv2
import torch

from run_c_study import (
    AUGMENTATIONS, MODES, PROJECT, environment_info, file_sha256, pilot, prepare_data, run_matrix,
)
from cnn_robustness_study import INPUT_MODES, tensor_views, verify_completed_run
from resnet_robustness_study import POOLING_MODES


DEFAULT_WEIGHTS_PATH = PROJECT / "assets/weights/resnet18-f37072fd.pth"
OFFICIAL_WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
CODE_FILES = ("run_resnet_study.py", "resnet_robustness_study.py", "run_c_study.py",
              "cnn_robustness_study.py", "cnn_robustness_pilot.py")
EVALUATIONS = ("train_clean", "validation_clean", "validation_motion", "validation_noise",
               "validation_dark", "validation_jpeg")
DATA_MANIFEST_KEYS = ("source_files", "generated_fingerprints", "extra_indices",
                      "augmentation_counts", "data_fingerprint", "preparation_seconds")


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def validate_resnet_config(config):
    """Validate recorded official-study metadata without loading models, weights or CUDA."""
    if not isinstance(config, dict):
        raise ValueError("ResNet configuration must be a JSON object")
    fixed = {
        "schema_version": 1, "classes": list(pilot.CLASSES),
        "backbone": "torchvision.models.resnet18", "feature_dim": 512,
        "network_input_size": 224, "image_size": 32, "pretrained": True,
        "weights_id": "ResNet18_Weights.IMAGENET1K_V1", "weights_sha256": OFFICIAL_WEIGHTS_SHA256,
        "batch_size": 128, "learning_rate": 0.003, "optimizer": "AdamW",
        "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
        "training_entries": 4410, "clean_training_images": 2940,
        "validation_images": 840, "train_tracks": 98, "validation_tracks": 28,
        "evaluation": "final_epoch_development_validation",
        "clahe": {"clip_limit": 2.0, "tile_grid": [4, 4], "channel": "LAB_L"},
        "denoising": {"method": "bilateral", "d": 5, "sigmaColor": 25, "sigmaSpace": 2},
    }
    for key, value in fixed.items():
        if config.get(key) != value or (key == "pretrained" and config.get(key) is not True):
            raise ValueError(f"Not the official ResNet18 study protocol: {key}")
    for key in ("epochs", "threads"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid ResNet study {key}")
    if config.get("device") not in ("cpu", "cuda"):
        raise ValueError("Recorded ResNet study device must be cpu or cuda")
    if config.get("pooling", "gap") not in POOLING_MODES:
        raise ValueError("Invalid ResNet pooling mode")
    for key in ("weights_path", "torchvision_version", "torchvision_module_path"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing ResNet provenance: {key}")
    weights_path = PurePosixPath(config["weights_path"])
    if (weights_path.is_absolute() or PureWindowsPath(config["weights_path"]).drive
            or ".." in weights_path.parts or "\\" in config["weights_path"]):
        raise ValueError("weights_path must be a portable project-relative path")
    for key in ("data_fingerprint", "split_sha256"):
        if not _is_sha256(config.get(key)):
            raise ValueError(f"Missing or invalid ResNet fingerprint: {key}")
    fingerprints = config.get("code_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != set(CODE_FILES) or any(
            not _is_sha256(value) for value in fingerprints.values()):
        raise ValueError("ResNet study must fingerprint its two new and three reused source files")
    environment = config.get("environment")
    if not isinstance(environment, dict) or any(key not in environment for key in (
            "python", "torch", "numpy", "opencv", "sklearn", "platform", "processor", "cpu_count",
            "device", "cuda_runtime", "cuda_device")):
        raise ValueError("Missing original environment metadata")
    for key in ("torchvision_version", "torchvision_module_path"):
        if environment.get(key) != config[key]:
            raise ValueError(f"Inconsistent torchvision provenance: {key}")
    if environment.get("torchvision") != config["torchvision_version"]:
        raise ValueError("Missing or inconsistent torchvision environment version")
    if environment["device"] != config["device"]:
        raise ValueError("Recorded environment and training device differ")
    return True


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid or incomplete study metadata: {path}: {error}") from error


def _code_fingerprints():
    return {name: file_sha256(Path(__file__).parent / name) for name in CODE_FILES}


def _is_spark_reservation(output_dir):
    """Recognize only the runtime-only directory reserved by the Spark wrapper."""
    try:
        entries = list(Path(output_dir).iterdir())
        if len(entries) != 1:
            return False
        runtime = entries[0]
        if runtime.name != "spark_runtime.json" or runtime.is_symlink() or not runtime.is_file():
            return False
        record = _read_json(runtime)
        return (isinstance(record, dict) and type(record.get("schema_version")) is int
                and record["schema_version"] == 1 and isinstance(record.get("image_id"), str)
                and record["image_id"].startswith("sha256:") and _is_sha256(record["image_id"][7:]))
    except (OSError, ValueError):
        return False


def _initialize_study(output_dir, study_config, manifest, split_path, spark_reserved=False):
    """Create study metadata exclusively, preserving a valid Spark runtime reservation."""
    output_dir = Path(output_dir)
    if spark_reserved:
        # Recheck after data preparation so files added meanwhile cannot be overwritten.
        if not _is_spark_reservation(output_dir):
            raise ValueError("Spark reservation is invalid or contains partial study files")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "run_config.json").open("x", encoding="utf-8") as handle:
        json.dump(study_config, handle, indent=2, ensure_ascii=False)
    with Path(split_path).open("rb") as source, (output_dir / "split.csv").open("xb") as target:
        shutil.copyfileobj(source, target)
    with (output_dir / "data_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _load_study(output_dir, matrix, epochs, pooling=None):
    """Read and validate study-wide provenance before verification or resumed training."""
    study = _read_json(output_dir / "run_config.json")
    if not isinstance(study, dict) or set(study) != {"common", "matrix"}:
        raise ValueError("Invalid study configuration envelope")
    validate_resnet_config(study["common"])
    if study["matrix"] != [list(row) for row in matrix] or study["common"]["epochs"] != epochs:
        raise ValueError("Existing study does not match the requested matrix and epochs")
    if pooling is not None and study["common"].get("pooling", "gap") != pooling:
        raise ValueError("Existing study uses a different pooling mode")
    if study["common"]["code_fingerprints"] != _code_fingerprints():
        raise ValueError("Study code fingerprints differ from the current source files")
    manifest = _read_json(output_dir / "data_manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != set(DATA_MANIFEST_KEYS):
        raise ValueError("Incomplete study data manifest")
    if manifest["data_fingerprint"] != study["common"]["data_fingerprint"]:
        raise ValueError("Study data fingerprints disagree")
    try:
        split_sha256 = file_sha256(output_dir / "split.csv")
    except OSError as error:
        raise ValueError("Missing study split.csv") from error
    if split_sha256 != study["common"]["split_sha256"]:
        raise ValueError("Study split.csv fingerprint differs")
    expected_ids = {f"{augmentation}_{mode}_seed{seed}" for augmentation, mode, seed in matrix}
    runs_root = output_dir / "runs"
    if runs_root.exists() and any(not path.is_dir() or path.name not in expected_ids
                                  for path in runs_root.iterdir()):
        raise ValueError("Unexpected run directory or file outside the requested matrix")
    return study, manifest


def _verify_run(run_dir, expected):
    saved = _read_json(run_dir / "config.json")
    validate_resnet_config(saved)
    comparable = {key: value for key, value in saved.items() if key != "preprocessing_seconds"}
    if comparable != expected:
        raise ValueError(f"Existing run configuration differs: {run_dir.name}")
    verify_completed_run(run_dir, saved)
    completion = _read_json(run_dir / "completion.json")
    if completion.get("evaluations") != list(EVALUATIONS):
        raise ValueError(f"Run must export all six frozen evaluations: {run_dir.name}")
    if completion.get("fusion") != INPUT_MODES[saved["input"]][0]:
        raise ValueError(f"Completion fusion does not match input mode: {run_dir.name}")
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/GTSRB/raw")
    parser.add_argument("--split", type=Path, default=PROJECT / "results/pilot_split.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/resnet_study_v1")
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--inputs", nargs="+", choices=MODES)
    parser.add_argument("--augmentations", nargs="+", choices=AUGMENTATIONS)
    parser.add_argument("--pooling", choices=POOLING_MODES,
                        help="Global pooling (default GAP, or the saved mode on resume)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs/threads must be positive and seeds unique")
    for values in (args.inputs, args.augmentations):
        if values and len(set(values)) != len(values):
            parser.error("duplicate inputs or augmentations")
    matrix = run_matrix(args.inputs, args.augmentations, args.seeds)
    if args.verify_only:
        study, _ = _load_study(args.output_dir, matrix, args.epochs, pooling=args.pooling)
        for augmentation, mode, seed in matrix:
            run_id = f"{augmentation}_{mode}_seed{seed}"
            expected = dict(study["common"], run_id=run_id, seed=seed, input=mode, augmentation=augmentation)
            _verify_run(args.output_dir / "runs" / run_id, expected)
        print(f"Verified {len(matrix)} completed official ResNet18 runs", flush=True)
        return
    existing_study, spark_reserved = None, False
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists; choose a new directory or --resume: {args.output_dir}")
        spark_reserved = _is_spark_reservation(args.output_dir)
        if not spark_reserved:
            existing_study, existing_manifest = _load_study(
                args.output_dir, matrix, args.epochs, pooling=args.pooling
            )
            # Refuse partial runs before any expensive preparation or potential new writes.
            for augmentation, mode, seed in matrix:
                run_id = f"{augmentation}_{mode}_seed{seed}"
                directory = args.output_dir / "runs" / run_id
                if directory.exists():
                    expected = dict(existing_study["common"], run_id=run_id, seed=seed,
                                    input=mode, augmentation=augmentation)
                    _verify_run(directory, expected)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this PyTorch installation")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    # Resolving torchvision and the pretrained checkpoint happens only for real training.
    from resnet_robustness_study import fit_run, weights_metadata
    metadata = weights_metadata(pretrained=True, weights_path=args.weights_path)
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print("Preparing frozen data and checking ResNet study fingerprints...", flush=True)
    data = prepare_data(args.data_root, args.split)
    env = environment_info(device)
    env.update(torchvision=metadata["torchvision_version"],
               torchvision_version=metadata["torchvision_version"],
               torchvision_module_path=metadata["torchvision_module_path"])
    common = {
        "schema_version": 1, "classes": list(pilot.CLASSES), "epochs": args.epochs,
        "batch_size": 128, "learning_rate": 0.003, "optimizer": "AdamW",
        "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
        "image_size": 32, "threads": args.threads, "device": device,
        "training_entries": 4410, "clean_training_images": 2940,
        "validation_images": 840, "train_tracks": 98, "validation_tracks": 28,
        "data_fingerprint": data["data_fingerprint"], "split_sha256": file_sha256(args.split),
        "code_fingerprints": _code_fingerprints(), "environment": env,
        "evaluation": "final_epoch_development_validation",
        "clahe": {"clip_limit": 2.0, "tile_grid": [4, 4], "channel": "LAB_L"},
        "denoising": {"method": "bilateral", "d": 5, "sigmaColor": 25, "sigmaSpace": 2},
        **metadata,
    }
    pooling = (existing_study["common"].get("pooling", "gap")
               if existing_study is not None else args.pooling or "gap")
    if pooling != "gap":
        common["pooling"] = pooling
    validate_resnet_config(common)
    study_config = {"common": common, "matrix": [list(row) for row in matrix]}
    manifest = {key: data[key] for key in DATA_MANIFEST_KEYS}
    if existing_study is not None:
        if existing_study != study_config:
            raise ValueError("Existing study configuration differs; use a new output directory")
        stable = lambda value: {key: item for key, item in value.items() if key != "preparation_seconds"}
        if stable(existing_manifest) != stable(manifest):
            raise ValueError("Existing study data manifest differs; use a new output directory")
    else:
        _initialize_study(args.output_dir, study_config, manifest, args.split, spark_reserved=spark_reserved)
    # Cache 32x32 CPU RGB views only; the model applies official 224 preprocessing per batch.
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
        train_views = None
        for _, mode, seed in (row for row in matrix if row[0] == augmentation):
            run_id = f"{augmentation}_{mode}_seed{seed}"
            run_dir = args.output_dir / "runs" / run_id
            config = dict(common, run_id=run_id, seed=seed, input=mode, augmentation=augmentation)
            if run_dir.exists():
                _verify_run(run_dir, config)
                print(f"Verified and skipped {run_id}", flush=True)
                continue
            if train_views is None:
                started = time.perf_counter()
                train_views = tensor_views(data["training"][augmentation])
                preprocess[f"training_{augmentation}"] = time.perf_counter() - started
            config["preprocessing_seconds"] = dict(preprocess, data_preparation=data["preparation_seconds"])
            print(f"Running {run_id} ({args.epochs} epochs, {device}, official ResNet18)", flush=True)
            fit_run(run_dir, config, train_views, train_targets, evaluations)
            _verify_run(run_dir, {key: value for key, value in config.items() if key != "preprocessing_seconds"})
        del train_views
    print(f"Completed and verified {len(matrix)} official ResNet18 runs in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
