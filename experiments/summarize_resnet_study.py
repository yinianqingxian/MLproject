"""
用途：核验并汇总官方预训练 ResNet18 的十五次融合实验，生成独立开发验证报告。
核心职责：
- 拒绝历史骨干与错误预训练来源，检查配置矩阵、512 维特征对齐和独立复算指标。
- 汇总三种子波动、配对差值、门控干预、训练曲线和计算成本。
- 输出中英文表格、出版用图和中文报告，并明确开发验证的证据边界。
关键副作用/安全约束：
- 仅读取每次运行的原始产物；全部核验通过后重建汇总文件和报告。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKBONE = "torchvision.models.resnet18"
WEIGHTS_ID = "ResNet18_Weights.IMAGENET1K_V1"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
FEATURE_DIM = 512
NETWORK_INPUT_SIZE = 224
CLASSES = (1, 2, 3, 4, 5, 7, 8)
SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "motion", "noise", "dark", "jpeg")
AUGMENTATIONS = ("repeat_clean", "targeted_motion", "mixed")
CONFIGURATIONS = tuple((aug, "dual") for aug in AUGMENTATIONS) + (
    ("mixed", "fixed"), ("mixed", "dual_denoised"),
)
METRIC_FIELDS = (
    "run_id", "seed", "augmentation", "input", "split", "condition", "severity",
    "macro_f1", "accuracy", "n_images", "parameters",
)
PREDICTION_FIELDS = (
    "run_id", "seed", "split", "image", "track_id", "condition", "severity",
    "y_true", "y_pred", "gate",
)
LABELS = {
    ("repeat_clean", "dual"): "Repeated clean / gated",
    ("targeted_motion", "dual"): "Motion training / gated",
    ("mixed", "dual"): "Mixed / gated",
    ("mixed", "fixed"): "Mixed / trained fixed mean",
    ("mixed", "dual_denoised"): "Mixed / denoised gated",
}
SHORT_LABELS = {
    ("repeat_clean", "dual"): "A0 gated",
    ("targeted_motion", "dual"): "A1 gated",
    ("mixed", "dual"): "A2 gated",
    ("mixed", "fixed"): "A2 fixed",
    ("mixed", "dual_denoised"): "A2 denoised",
}
COMPARISONS = (
    ("targeted_minus_repeat", ("targeted_motion", "dual"), ("repeat_clean", "dual")),
    ("mixed_minus_repeat", ("mixed", "dual"), ("repeat_clean", "dual")),
    ("mixed_minus_targeted", ("mixed", "dual"), ("targeted_motion", "dual")),
    ("trained_fixed_minus_gated", ("mixed", "fixed"), ("mixed", "dual")),
    ("denoised_minus_gated", ("mixed", "dual_denoised"), ("mixed", "dual")),
)
COMPARISON_LABELS = {
    "targeted_minus_repeat": "Motion - repeated clean",
    "mixed_minus_repeat": "Mixed - repeated clean",
    "mixed_minus_targeted": "Mixed - motion",
    "trained_fixed_minus_gated": "Trained fixed - gated (mixed)",
    "denoised_minus_gated": "Denoised - gated (mixed)",
}


class StudyValidationError(ValueError):
    """A source artifact violates the frozen C-study protocol."""


@dataclass
class Run:
    directory: Path
    config: dict[str, Any]
    metrics: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    interventions: list[dict[str, Any]]
    intervention_predictions: list[dict[str, Any]]
    training: list[dict[str, Any]]
    cost: dict[str, Any]
    gates: list[dict[str, Any]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StudyValidationError(message)


def read_csv(path: Path, fields: Iterable[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing artifact: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(set(fields) <= set(reader.fieldnames or []), f"Invalid columns: {path}")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values())
                for row in rows), f"Malformed CSV row: {path}")
    return rows


def number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise StudyValidationError(f"{name} must be numeric: {value!r}") from error
    require(math.isfinite(result), f"{name} must be finite")
    return result


def integer(value: Any, name: str) -> int:
    numeric = number(value, name)
    require(numeric.is_integer(), f"{name} must be an integer")
    return int(numeric)


def classification_scores(y_true: Iterable[int], y_pred: Iterable[int]) -> tuple[float, float]:
    """Recompute seven-class macro F1 and accuracy without the trainer's metric code."""
    truth, prediction = np.asarray(list(y_true), dtype=int), np.asarray(list(y_pred), dtype=int)
    require(truth.shape == prediction.shape and truth.ndim == 1 and truth.size > 0,
            "Predictions and labels must be matching nonempty vectors")
    require(set(truth) <= set(CLASSES) and set(prediction) <= set(CLASSES), "Unknown class id")
    f1_values = []
    for class_id in CLASSES:
        true_positive = int(np.sum((truth == class_id) & (prediction == class_id)))
        denominator = int(np.sum(truth == class_id) + np.sum(prediction == class_id))
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(f1_values)), float(np.mean(truth == prediction))


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    require(len(array) > 0 and bool(np.isfinite(array).all()), "Cannot summarize empty/nonfinite data")
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _validate_backbone(config: dict[str, Any]) -> None:
    """Validate official study identity without constructing models or importing torchvision."""
    identity = {"backbone": BACKBONE, "feature_dim": FEATURE_DIM,
                "network_input_size": NETWORK_INPUT_SIZE, "image_size": 32,
                "weights_id": WEIGHTS_ID, "weights_sha256": WEIGHTS_SHA256}
    for key, expected in identity.items():
        require(config.get(key) == expected, f"Official ResNet18 protocol mismatch: {key} != {expected}")
    require(config.get("pretrained") is True, "Official ResNet18 study requires pretrained=True")
    version = config.get("torchvision_version")
    require(isinstance(version, str) and bool(version.strip()), "Missing torchvision_version")
    environment = config.get("environment", {})
    require(isinstance(environment, dict), "Invalid recorded environment")
    if "torchvision" in environment:
        require(environment["torchvision"] == version, "Recorded torchvision versions differ")
    weight_path = config.get("weights_path")
    require(isinstance(weight_path, str) and bool(weight_path), "Missing project-relative weights_path")
    portable = PurePosixPath(weight_path.replace("\\", "/"))
    require(not portable.is_absolute() and not PureWindowsPath(weight_path).is_absolute()
            and not PureWindowsPath(weight_path).drive and ".." not in portable.parts,
            "weights_path must stay within the project and be project-relative")
    # The runner owns the complete recorded environment and source schema. This
    # read-only validator does not import torchvision or resolve local GPU state.
    from run_resnet_study import validate_resnet_config
    try:
        validate_resnet_config(config)
    except ValueError as error:
        raise StudyValidationError(f"Official ResNet18 metadata invalid: {error}") from error


def _verified_weights_path(config: dict[str, Any]) -> Path:
    """Check the packaged official initialization, never downloading missing weights."""
    path = (PROJECT_ROOT / config["weights_path"]).resolve()
    require(path.is_relative_to(PROJECT_ROOT.resolve()) and path.is_file(),
            f"Missing project-contained official weights: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    require(digest.hexdigest() == config["weights_sha256"] == WEIGHTS_SHA256,
            f"Official ResNet18 initialization checksum mismatch: {path}")
    return path


def _validate_config(config: dict[str, Any], directory: Path) -> None:
    _validate_backbone(config)
    require(config.get("run_id") == directory.name, f"Run id/directory mismatch: {directory}")
    require((config.get("augmentation"), config.get("input")) in CONFIGURATIONS,
            f"Unexpected C configuration: {directory}")
    require(config.get("seed") in SEEDS, f"Unexpected seed: {directory}")
    require(tuple(config.get("classes", [])) == CLASSES, f"Class order mismatch: {directory}")
    protocol = {"epochs": 10, "batch_size": 128, "learning_rate": 0.003, "image_size": 32,
                "optimizer": "AdamW", "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8,
                "training_entries": 4410, "clean_training_images": 2940,
                "validation_images": 840, "train_tracks": 98, "validation_tracks": 28}
    for key, expected in protocol.items():
        require(config.get(key) == expected, f"Protocol mismatch {directory.name}: {key} != {expected}")
    require(config.get("clahe") == {"channel": "LAB_L", "clip_limit": 2.0, "tile_grid": [4, 4]},
            f"CLAHE protocol mismatch: {directory.name}")
    require(config.get("denoising") == {"method": "bilateral", "d": 5, "sigmaColor": 25, "sigmaSpace": 2},
            f"Denoising protocol mismatch: {directory.name}")
    require(bool(config.get("data_fingerprint")), f"Missing data fingerprint: {directory}")


def _validate_predictions(rows: list[dict[str, Any]], config: dict[str, Any], *, intervention: bool = False) -> None:
    expected_count = 12600 if intervention else 4200
    require(len(rows) == expected_count,
            f"{config['run_id']}: expected {expected_count} prediction rows, got {len(rows)}")
    observed = set()
    grouped = defaultdict(list)
    for row in rows:
        require(row["run_id"] == config["run_id"] and integer(row["seed"], "seed") == config["seed"],
                f"Prediction configuration mismatch: {config['run_id']}")
        require(row["split"] == "validation" and row["condition"] in CONDITIONS,
                f"Unexpected prediction condition/split: {config['run_id']}")
        row["seed"] = integer(row["seed"], "seed")
        for field in ("y_true", "y_pred"):
            row[field] = integer(row[field], field)
            require(row[field] in CLASSES, f"Invalid {field}: {config['run_id']}")
        row["gate"] = number(row["gate"], "gate")
        require(0 <= row["gate"] <= 1, f"Gate out of range: {config['run_id']}")
        require(bool(row["image"]) and bool(row["track_id"]), "Missing image or track")
        group = (row["condition"],)
        if intervention:
            row["gate_override"] = number(row["gate_override"], "gate_override")
            require(row["gate_override"] in (0.0, 0.5, 1.0), "Invalid gate override")
            require(abs(row["gate"] - row["gate_override"]) <= 1e-7, "Override/gate mismatch")
            group += (row["gate_override"],)
        elif config["input"] == "fixed":
            require(abs(row["gate"] - 0.5) <= 1e-7, "Trained-fixed gate must be 0.5")
        unique_key = group + (row["image"],)
        require(unique_key not in observed, f"Duplicate prediction: {unique_key}")
        observed.add(unique_key)
        grouped[group].append(row)
    require(len(grouped) == (15 if intervention else 5), "Incomplete prediction groups")
    reference = None
    for group, values in grouped.items():
        require(len(values) == 840, f"{group}: expected 840 validation frames")
        metadata = sorted((row["image"], row["track_id"], row["y_true"]) for row in values)
        if reference is None:
            reference = metadata
        require(metadata == reference, f"Validation metadata differs across conditions/overrides: {group}")


def _validate_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]],
                      config: dict[str, Any], *, intervention: bool = False) -> None:
    require(len(rows) == (15 if intervention else 5), f"Invalid metric count: {config['run_id']}")
    prediction_groups = defaultdict(list)
    for row in predictions:
        key = (row["condition"], float(row["gate_override"])) if intervention else (row["condition"],)
        prediction_groups[key].append(row)
    seen = set()
    for row in rows:
        for key in ("run_id", "augmentation", "input"):
            require(row[key] == config[key], f"Metric {key} mismatch: {config['run_id']}")
        row["seed"] = integer(row["seed"], "seed")
        require(row["seed"] == config["seed"] and row["split"] == "validation", "Metric split/seed mismatch")
        require(row["condition"] in CONDITIONS, "Unknown metric condition")
        row["n_images"], row["parameters"] = integer(row["n_images"], "n_images"), integer(row["parameters"], "parameters")
        require(row["n_images"] == 840 and row["parameters"] > 0, "Invalid metric image/parameter count")
        key = (row["condition"],)
        if intervention:
            row["gate_override"] = number(row["gate_override"], "gate_override")
            key += (row["gate_override"],)
        require(key not in seen and key in prediction_groups, "Duplicate/missing metric group")
        seen.add(key)
        values = prediction_groups[key]
        require({value["severity"] for value in values} == {row["severity"]}, "Severity mismatch")
        expected_f1, expected_accuracy = classification_scores(
            (value["y_true"] for value in values), (value["y_pred"] for value in values))
        for field, expected in (("macro_f1", expected_f1), ("accuracy", expected_accuracy)):
            row[field] = number(row[field], field)
            require(abs(row[field] - expected) <= 1e-8,
                    f"Metric recomputation mismatch {config['run_id']} {key} {field}: {row[field]} != {expected}")


def _validate_metadata(metadata: list[tuple[str, str, int]], split: str) -> None:
    expected_images, expected_tracks, per_class_tracks = (2940, 98, 14) if split == "train" else (840, 28, 4)
    require(len(metadata) == expected_images, f"Invalid {split} image count")
    images = [row[0] for row in metadata]
    require(len(set(images)) == expected_images and images == sorted(images), f"{split} images not unique/sorted")
    track_counts = Counter(row[1] for row in metadata)
    require(len(track_counts) == expected_tracks and set(track_counts.values()) == {30}, f"Invalid {split} tracks")
    class_tracks = defaultdict(set)
    track_labels = defaultdict(set)
    for _, track, label in metadata:
        require(label in CLASSES and bool(track), f"Invalid {split} label/track")
        class_tracks[label].add(track)
        track_labels[track].add(label)
    require(set(class_tracks) == set(CLASSES) and all(len(value) == per_class_tracks for value in class_tracks.values()),
            f"Invalid per-class {split} track counts")
    require(all(len(value) == 1 for value in track_labels.values()), "Track contains multiple labels")


def _gate_summary(config: dict[str, Any], split: str, condition: str,
                  gate: np.ndarray, metadata: list[tuple[str, str, int]], label: int | str) -> dict[str, Any]:
    selected = np.array([True if label == "all" else row[2] == label for row in metadata])
    values = gate[selected]
    mean, sd = mean_sd(values)
    quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return {"run_id": config["run_id"], "seed": config["seed"], "augmentation": config["augmentation"],
            "input": config["input"], "split": split, "condition": condition, "class_id": label,
            "n_images": len(values), "n_tracks": len({row[1] for row, keep in zip(metadata, selected) if keep}),
            "gate_mean": mean, "gate_sd": sd,
            "fraction_gate_ge_099": float(np.mean(values >= .99)),
            "fraction_gate_exactly_one": float(np.mean(values == 1)),
            **dict(zip(("minimum", "q05", "q25", "median", "q75", "q95", "maximum"), map(float, quantiles)))}


def _validate_features(run_dir: Path, config: dict[str, Any], predictions: list[dict[str, Any]],
                       reference_split: dict[str, list[tuple[str, str, int]]]) -> list[dict[str, Any]]:
    expected_names = {"train_clean"} | {f"validation_{condition}" for condition in CONDITIONS}
    feature_dir = run_dir / "features"
    require({path.stem for path in feature_dir.glob("*.npz")} == expected_names,
            f"Expected exactly six feature archives: {run_dir}")
    gate_summaries = []
    split_metadata = {}
    for name in sorted(expected_names):
        split, condition = name.split("_", 1)
        with np.load(feature_dir / f"{name}.npz", allow_pickle=False) as archive:
            require({"image", "track_id", "y_true", "y_pred", "features", "gate"} <= set(archive.files),
                    f"Feature schema mismatch: {name}")
            images, tracks = archive["image"].astype(str), archive["track_id"].astype(str)
            truth, predicted = archive["y_true"], archive["y_pred"]
            features, gate = archive["features"], archive["gate"].reshape(-1)
            count = 2940 if split == "train" else 840
            require(images.shape == tracks.shape == truth.shape == predicted.shape == gate.shape == (count,),
                    f"Feature metadata shape mismatch: {run_dir.name}/{name}")
            require(features.shape == (count, FEATURE_DIM) and bool(np.isfinite(features).all()),
                    f"Invalid 512-dimensional features: {run_dir.name}/{name}")
            require(bool(np.isfinite(gate).all()) and bool(((0 <= gate) & (gate <= 1)).all()), "Invalid feature gates")
            require(set(truth.tolist()) <= set(CLASSES) and set(predicted.tolist()) <= set(CLASSES), "Invalid feature labels")
            metadata = [(str(image), str(track), int(label)) for image, track, label in zip(images, tracks, truth)]
            _validate_metadata(metadata, split)
            require(metadata == reference_split[split], f"Feature metadata differs from frozen split: {run_dir.name}/{name}")
            split_metadata[split] = metadata
            if split == "validation":
                rows = sorted((row for row in predictions if row["condition"] == condition), key=lambda row: row["image"])
                require(metadata == [(row["image"], row["track_id"], row["y_true"]) for row in rows], "NPZ/CSV metadata mismatch")
                require(predicted.tolist() == [row["y_pred"] for row in rows], "NPZ/CSV predictions differ")
                require(bool(np.allclose(gate, [row["gate"] for row in rows], atol=1e-7, rtol=0)), "NPZ/CSV gates differ")
            if config["input"] == "fixed":
                require(bool(np.allclose(gate, .5, atol=1e-7, rtol=0)), "Fixed feature gate differs from 0.5")
            gate_summaries.extend(_gate_summary(config, split, condition, gate, metadata, label)
                                  for label in ("all", *CLASSES))
    require(not ({row[1] for row in split_metadata["train"]} & {row[1] for row in split_metadata["validation"]}),
            "Training/validation track leakage")
    return gate_summaries


def _reference_split(study_dir: Path) -> dict[str, list[tuple[str, str, int]]]:
    candidates = (study_dir / "split.csv", study_dir / "pilot_split.csv", PROJECT_ROOT / "results" / "pilot_split.csv")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    require(path is not None, "Cannot find frozen split.csv or results/pilot_split.csv")
    rows = read_csv(path, ("split", "image", "class_id", "track_id"))
    require(len(rows) == 3780, "Frozen split must contain 3780 clean images")
    result = {split: sorted((row["image"], row["track_id"], integer(row["class_id"], "class_id"))
                            for row in rows if row["split"] == split) for split in ("train", "validation")}
    for split, metadata in result.items():
        _validate_metadata(metadata, split)
    require(not ({row[1] for row in result["train"]} & {row[1] for row in result["validation"]}), "Frozen split has track leakage")
    return result


def _validate_study_metadata(study_dir: Path, common: dict[str, Any]) -> dict[str, list[tuple[str, str, int]]]:
    """Validate the frozen data snapshot without reopening source images or current training code."""
    from run_resnet_study import DATA_MANIFEST_KEYS

    manifest_path = study_dir / "data_manifest.json"
    require(manifest_path.is_file(), f"Missing study data_manifest.json: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StudyValidationError(f"Invalid study data manifest: {manifest_path}") from error
    require(isinstance(manifest, dict) and set(manifest) == set(DATA_MANIFEST_KEYS),
            "Incomplete study data manifest schema")
    require(manifest["data_fingerprint"] == common["data_fingerprint"], "Study data fingerprints disagree")

    split_path = study_dir / "split.csv"
    require(split_path.is_file(), f"Missing study split.csv: {split_path}")
    require(hashlib.sha256(split_path.read_bytes()).hexdigest() == common["split_sha256"],
            "Study split.csv fingerprint differs from the frozen configuration")
    rows = read_csv(split_path, ("split", "image", "class_id", "track_id"))
    ordered_split = [(row["split"], row["image"], integer(row["class_id"], "class_id"), row["track_id"])
                     for row in rows]
    reference = _reference_split(study_dir)

    def is_sha256(value: Any) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)

    sources = manifest["source_files"]
    require(isinstance(sources, list) and len(sources) == 3780 and
            all(isinstance(row, dict) and set(row) == {"image", "sha256"} and
                isinstance(row["image"], str) and is_sha256(row["sha256"]) for row in sources),
            "Invalid data manifest source_files")
    require([row["image"] for row in sources] == [row["image"] for row in rows],
            "Data manifest sources differ from the ordered study split")
    generated = manifest["generated_fingerprints"]
    expected_generated = {f"training/{augmentation}" for augmentation in AUGMENTATIONS} | {
        f"validation/{condition}" for condition in CONDITIONS}
    require(isinstance(generated, dict) and set(generated) == expected_generated and
            all(is_sha256(value) for value in generated.values()),
            "Invalid data manifest generated_fingerprints")
    extras = manifest["extra_indices"]
    require(isinstance(extras, list) and len(extras) == 1470 and
            all(type(index) is int and 0 <= index < 2940 for index in extras) and len(set(extras)) == 1470,
            "Invalid data manifest extra_indices")
    require(manifest["augmentation_counts"] == {
        "repeat_clean": {"repeat": 1470}, "targeted_motion": {"motion": 1470},
        "mixed": {"motion": 490, "noise": 490, "dark": 490}}, "Invalid data manifest augmentation_counts")
    require(number(manifest["preparation_seconds"], "data_manifest.preparation_seconds") >= 0,
            "Negative data manifest preparation_seconds")
    # This is prepare_data's stable snapshot identity: recorded source hashes,
    # original split order, then training and validation pixel hashes in order.
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode())
    digest.update(json.dumps(ordered_split, separators=(",", ":")).encode())
    for name, fingerprint in sorted(generated.items()):
        digest.update(f"{name}:{fingerprint}".encode())
    require(digest.hexdigest() == common["data_fingerprint"],
            "Study data fingerprint does not match the recorded source/generated snapshot")
    return reference


def load_study(study_dir: Path) -> list[Run]:
    """Validate every source before returning any aggregate-ready observations."""
    runs_dir = study_dir / "runs"
    require(runs_dir.is_dir(), f"Missing runs directory: {runs_dir}")
    directories = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    require(len(directories) == 15, f"Expected exactly 15 run directories; found {len(directories)}. No partial report written.")
    manifest_path = study_dir / "run_config.json"
    require(manifest_path.is_file(), f"Missing frozen study manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        study_config = json.load(handle)
    require(isinstance(study_config, dict) and set(study_config) == {"common", "matrix"},
            "Study manifest must contain exactly common and matrix")
    common = study_config["common"]
    frozen_matrix = study_config["matrix"]
    require(isinstance(common, dict), "Study common configuration must be an object")
    _validate_backbone(common)
    require(isinstance(frozen_matrix, list) and len(frozen_matrix) == 15 and
            all(isinstance(row, list) and len(row) == 3 and isinstance(row[0], str)
                and isinstance(row[1], str) and isinstance(row[2], int) for row in frozen_matrix),
            "Invalid frozen study matrix schema")
    expected_matrix = {(aug, mode, seed) for aug, mode in CONFIGURATIONS for seed in SEEDS}
    require(set(map(tuple, frozen_matrix)) == expected_matrix,
            "Frozen study matrix must contain exactly the 15 C configurations")
    reference = _validate_study_metadata(study_dir, common)
    prepared = []
    matrix = set()
    fingerprints = set()
    for directory in directories:
        with (directory / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        _validate_config(config, directory)
        per_run_fields = {"run_id", "seed", "input", "augmentation", "preprocessing_seconds"}
        require(per_run_fields <= set(config), f"Missing per-run configuration fields: {directory.name}")
        observed_common = {key: value for key, value in config.items() if key not in per_run_fields}
        require(observed_common == common, f"Frozen study common configuration mismatch: {directory.name}")
        matrix_key = (config["augmentation"], config["input"], config["seed"])
        require(matrix_key not in matrix, f"Duplicate configuration: {matrix_key}")
        matrix.add(matrix_key)
        fingerprints.add(json.dumps(config["data_fingerprint"], sort_keys=True))
        prepared.append((directory, config))
    require(matrix == expected_matrix, "C matrix incomplete")
    require(len(fingerprints) == 1, "Runs do not share a common data fingerprint")
    _verified_weights_path(common)
    from cnn_robustness_study import verify_completed_run

    result = []
    for directory, config in prepared:
        try:
            verified = verify_completed_run(directory, config)
        except ValueError as error:
            raise StudyValidationError(f"Trainer integrity check failed: {directory}: {error}") from error
        require(bool(verified), f"Trainer integrity check failed: {directory}")
        metrics = read_csv(directory / "metrics.csv", METRIC_FIELDS)
        predictions = read_csv(directory / "predictions.csv", PREDICTION_FIELDS)
        _validate_predictions(predictions, config)
        _validate_metrics(metrics, predictions, config)
        gate_summaries = _validate_features(directory, config, predictions, reference)
        interventions, intervention_predictions = [], []
        if config["input"] != "fixed":
            interventions = read_csv(directory / "intervention_metrics.csv", (*METRIC_FIELDS, "gate_override"))
            intervention_predictions = read_csv(directory / "intervention_predictions.csv", (*PREDICTION_FIELDS, "gate_override"))
            _validate_predictions(intervention_predictions, config, intervention=True)
            _validate_metrics(interventions, intervention_predictions, config, intervention=True)
            natural_metadata = {(row["condition"], row["image"]): (row["track_id"], row["y_true"], row["severity"])
                                for row in predictions}
            require(all(natural_metadata[(row["condition"], row["image"])] ==
                        (row["track_id"], row["y_true"], row["severity"]) for row in intervention_predictions),
                    "Intervention/natural metadata mismatch")
        else:
            require(not (directory / "intervention_metrics.csv").exists() and
                    not (directory / "intervention_predictions.csv").exists(),
                    "Trained-fixed model must not be presented as a learned-gate intervention")
        training = read_csv(directory / "training_log.csv", ("epoch", "train_loss", "validation_macro_f1", "validation_accuracy", "epoch_seconds"))
        require(len(training) == 10 and [integer(row["epoch"], "epoch") for row in training] == list(range(1, 11)),
                f"Invalid training epochs: {directory.name}")
        for row in training:
            row["epoch"] = integer(row["epoch"], "epoch")
            for key in ("train_loss", "validation_macro_f1", "validation_accuracy", "epoch_seconds"):
                row[key] = number(row[key], key)
                require(row[key] >= 0, f"Negative training quantity: {key}")
            require(row["validation_macro_f1"] <= 1 and row["validation_accuracy"] <= 1, "Training metric outside [0,1]")
        clean = next(row for row in metrics if row["condition"] == "clean")
        require(abs(training[-1]["validation_macro_f1"] - clean["macro_f1"]) <= 1e-8 and
                abs(training[-1]["validation_accuracy"] - clean["accuracy"]) <= 1e-8,
                "Final-epoch validation/export metrics differ")
        with (directory / "cost.json").open(encoding="utf-8") as handle:
            cost = json.load(handle)
        for key in ("parameters", "training_seconds", "export_seconds", "training_entries", "optimizer_steps", "threads"):
            cost[key] = number(cost.get(key), f"cost.{key}")
            require(cost[key] > 0, f"Invalid cost value {key}")
        require(cost["parameters"] == clean["parameters"] and all(row["parameters"] == clean["parameters"] for row in metrics + interventions),
                "Parameter count differs across artifacts")
        require(cost["training_entries"] == 4410 and cost["optimizer_steps"] == 350, "Unequal training budget")
        require(cost["threads"] == config["threads"], "Cost/config thread count differs")
        inference = cost.get("inference_seconds", {})
        require({"train_clean", *(f"validation_{condition}" for condition in CONDITIONS)} <= set(inference), "Missing inference timings")
        for name, value in inference.items():
            require(number(value, f"inference.{name}") >= 0, "Negative inference timing")
        result.append(Run(directory, config, metrics, predictions, interventions, intervention_predictions,
                          training, cost, gate_summaries))
    return result


def summarize_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row["augmentation"], row["input"], row["condition"], row["severity"])].append(row)
    result = []
    for augmentation, mode in CONFIGURATIONS:
        for condition in CONDITIONS:
            keys = [key for key in grouped if key[:3] == (augmentation, mode, condition)]
            require(len(keys) == 1, "Severity differs across seeds")
            key = keys[0]
            values = grouped[key]
            require({row["seed"] for row in values} == set(SEEDS) and len(values) == 3, "Missing summary seed")
            row = dict(zip(("augmentation", "input", "condition", "severity"), key))
            row.update(n_seeds=3, n_images_per_seed=840, n_independent_validation_tracks=28)
            for metric in ("macro_f1", "accuracy"):
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(value[metric] for value in values)
            result.append(row)
    return result


def paired_differences(metrics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(row["augmentation"], row["input"], row["seed"], row["condition"]): row for row in metrics}
    differences, summary = [], []
    for name, candidate, reference in COMPARISONS:
        for condition in CONDITIONS:
            values = []
            for seed in SEEDS:
                left, right = lookup[(*candidate, seed, condition)], lookup[(*reference, seed, condition)]
                require(left["severity"] == right["severity"], "Paired severities differ")
                row = {"comparison": name, "candidate_run_id": left["run_id"], "reference_run_id": right["run_id"],
                       "seed": seed, "condition": condition, "severity": left["severity"],
                       "macro_f1_delta_pp": 100 * (left["macro_f1"] - right["macro_f1"]),
                       "accuracy_delta_pp": 100 * (left["accuracy"] - right["accuracy"])}
                differences.append(row)
                values.append(row)
            row = {"comparison": name, "condition": condition, "severity": values[0]["severity"], "n_seeds": 3}
            for metric in ("macro_f1_delta_pp", "accuracy_delta_pp"):
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(value[metric] for value in values)
            row["positive_macro_f1_seeds"] = sum(value["macro_f1_delta_pp"] > 0 for value in values)
            row["negative_macro_f1_seeds"] = sum(value["macro_f1_delta_pp"] < 0 for value in values)
            summary.append(row)
    return differences, summary


def summarize_interventions(runs: list[Run]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    differences = []
    for run in runs:
        learned = {row["condition"]: row for row in run.metrics}
        for row in run.interventions:
            reference = learned[row["condition"]]
            differences.append({**row, "macro_f1_delta_pp": 100 * (row["macro_f1"] - reference["macro_f1"]),
                                "accuracy_delta_pp": 100 * (row["accuracy"] - reference["accuracy"])})
    grouped = defaultdict(list)
    for row in differences:
        grouped[(row["augmentation"], row["input"], row["condition"], row["gate_override"])].append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        require(len(values) == 3 and {row["seed"] for row in values} == set(SEEDS), "Missing intervention seed")
        row = dict(zip(("augmentation", "input", "condition", "gate_override"), key))
        row["n_seeds"] = 3
        for metric in ("macro_f1", "accuracy", "macro_f1_delta_pp", "accuracy_delta_pp"):
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(value[metric] for value in values)
        summary.append(row)
    return differences, summary


def convergence_and_cost(runs: list[Run]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    training, convergence, costs, inference = [], [], [], []
    for run in runs:
        meta = {key: run.config[key] for key in ("run_id", "seed", "augmentation", "input")}
        training.extend({**meta, **row} for row in run.training)
        first, last = run.training[0], run.training[-1]
        best = max(run.training, key=lambda row: row["validation_macro_f1"])
        convergence.append({**meta, "epochs": 10, "initial_train_loss": first["train_loss"],
                            "final_train_loss": last["train_loss"],
                            "train_loss_change_last_epoch": last["train_loss"] - run.training[-2]["train_loss"],
                            "final_validation_macro_f1": last["validation_macro_f1"],
                            "validation_change_last_epoch_pp": 100 * (last["validation_macro_f1"] - run.training[-2]["validation_macro_f1"]),
                            "best_validation_epoch_diagnostic_only": best["epoch"],
                            "final_minus_best_validation_pp": 100 * (last["validation_macro_f1"] - best["validation_macro_f1"]),
                            "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in run.training]))})
        cost = {**meta, **{key: run.cost[key] for key in
                         ("parameters", "training_seconds", "export_seconds", "training_entries", "optimizer_steps", "threads")}}
        cost["clean_inference_ms_per_image"] = 1000 * run.cost["inference_seconds"]["validation_clean"] / 840
        preprocessing = run.config.get("preprocessing_seconds", {})
        for name, seconds in sorted(preprocessing.items()):
            cost[f"preprocessing_{name}_seconds"] = number(seconds, f"preprocessing.{name}")
        costs.append(cost)
        for evaluation, seconds in sorted(run.cost["inference_seconds"].items()):
            count = 2940 if evaluation == "train_clean" else 840
            inference.append({**meta, "evaluation": evaluation, "n_images": count,
                              "inference_seconds": seconds, "inference_ms_per_image": 1000 * seconds / count})
    return training, convergence, costs, inference


def summarize_costs(costs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("parameters", "training_seconds", "export_seconds", "clean_inference_ms_per_image")
    result = []
    for augmentation, mode in CONFIGURATIONS:
        values = [row for row in costs if (row["augmentation"], row["input"]) == (augmentation, mode)]
        row = {"augmentation": augmentation, "input": mode, "n_seeds": 3,
               "threads": ",".join(str(int(value)) for value in sorted({row["threads"] for row in values}))}
        for field in fields:
            row[f"{field}_mean"], row[f"{field}_sd"] = mean_sd(value[field] for value in values)
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields) if fields else list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def make_figures(report_dir: Path, runs: list[Run], summary: list[dict[str, Any]],
                 paired: list[dict[str, Any]], paired_summary: list[dict[str, Any]],
                 interventions: list[dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "svg.fonttype": "none",
                         "savefig.dpi": 220, "figure.facecolor": "white"})
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = []
    palette = ("#4C78A8", "#E69F00", "#009E73", "#CC79A7", "#D55E00")

    def save(fig: Any, name: str) -> None:
        fig.savefig(figure_dir / f"{name}.png", bbox_inches="tight")
        fig.savefig(figure_dir / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        names.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3), sharex=True)
    lookup = {(row["augmentation"], row["input"], row["condition"]): row for row in summary}
    x = np.arange(len(CONDITIONS))
    for index, configuration in enumerate(CONFIGURATIONS):
        rows = [lookup[(*configuration, condition)] for condition in CONDITIONS]
        for ax, metric, label in zip(axes, ("macro_f1", "accuracy"), ("Macro F1 (%)", "Accuracy (%)")):
            offset = (index - 2) * .055
            ax.errorbar(x + offset, [100 * row[f"{metric}_mean"] for row in rows],
                        yerr=[100 * row[f"{metric}_sd"] for row in rows], marker="o", markersize=4,
                        capsize=3, linewidth=1.4, color=palette[index], label=LABELS[configuration])
            ax.set_ylabel(label)
            ax.set_xticks(x, [value.capitalize() for value in CONDITIONS])
            ax.grid(axis="y", alpha=.2)
    axes[0].set_title("Development validation: final epoch")
    axes[1].set_title("Mean and sample SD over three training seeds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(.5, -.07))
    fig.tight_layout(rect=(0, .12, 1, 1))
    save(fig, "scores_with_sd")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True)
    for index, (comparison, _, _) in enumerate(COMPARISONS):
        ax = axes.flat[index]
        rows = [next(row for row in paired_summary if row["comparison"] == comparison and row["condition"] == condition)
                for condition in CONDITIONS]
        ax.axhline(0, color=".5", linewidth=.8)
        for seed, offset in zip(SEEDS, (-.12, 0, .12)):
            values = [next(row["macro_f1_delta_pp"] for row in paired if row["comparison"] == comparison
                           and row["condition"] == condition and row["seed"] == seed) for condition in CONDITIONS]
            ax.scatter(x + offset, values, s=24, alpha=.65, label=f"Seed {seed}")
        ax.errorbar(x, [row["macro_f1_delta_pp_mean"] for row in rows],
                    yerr=[row["macro_f1_delta_pp_sd"] for row in rows], color="black", fmt="D", markersize=4,
                    capsize=4, label="Mean +/- SD")
        ax.set_title(COMPARISON_LABELS[comparison], fontsize=10)
        ax.set_ylabel("Macro F1 change (pp)")
        ax.set_xticks(x, [value.capitalize() for value in CONDITIONS], rotation=20)
        ax.grid(axis="y", alpha=.15)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("Paired changes using matching training seeds; n = 3")
    fig.tight_layout()
    save(fig, "paired_changes")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.3), sharex=True, sharey=True)
    learned_configurations = [value for value in CONFIGURATIONS if value[1] != "fixed"]
    for ax, configuration in zip(axes.flat, learned_configurations):
        grouped_gates = [[row["gate"] for run in runs if (run.config["augmentation"], run.config["input"]) == configuration
                          for row in run.predictions if row["condition"] == condition] for condition in CONDITIONS]
        plot = ax.boxplot(grouped_gates, positions=x, widths=.55, showfliers=False, whis=(5, 95), patch_artist=True)
        for box in plot["boxes"]:
            box.set(facecolor=palette[CONFIGURATIONS.index(configuration)], alpha=.3)
        for seed, offset in zip(SEEDS, (-.12, 0, .12)):
            run = next(run for run in runs if (run.config["augmentation"], run.config["input"], run.config["seed"]) == (*configuration, seed))
            means = [np.mean([row["gate"] for row in run.predictions if row["condition"] == condition]) for condition in CONDITIONS]
            ax.scatter(x + offset, means, s=21, color="black", alpha=.6)
        ax.set_title(LABELS[configuration])
        ax.set_ylabel("Enhanced-view weight g")
        ax.set_xticks(x, [value.capitalize() for value in CONDITIONS])
        ax.set_ylim(-.025, 1.025)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Gate distributions: pooled frames; dots are seed means\nBoxes: 25-75%; whiskers: 5-95%; descriptive only")
    fig.tight_layout()
    save(fig, "gate_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for configuration, color in zip(CONFIGURATIONS, palette):
        selected = [run for run in runs if (run.config["augmentation"], run.config["input"]) == configuration]
        epochs = np.arange(1, 11)
        for ax, metric in zip(axes, ("train_loss", "validation_macro_f1")):
            values = np.array([[row[metric] for row in run.training] for run in selected])
            scale = 100 if metric == "validation_macro_f1" else 1
            mean, sd = values.mean(axis=0) * scale, values.std(axis=0, ddof=1) * scale
            ax.plot(epochs, mean, label=LABELS[configuration], color=color, linewidth=1.6)
            ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=.11)
            ax.set_xlabel("Epoch")
            ax.set_xticks([1, 2, 4, 6, 8, 10])
            ax.grid(alpha=.15)
    axes[0].set_ylabel("Training cross-entropy loss")
    axes[1].set_ylabel("Clean validation macro F1 (%)")
    axes[0].set_title("Training trajectory")
    axes[1].set_title("Final epoch used for every comparison")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, bbox_to_anchor=(.5, -.07))
    fig.tight_layout(rect=(0, .12, 1, 1))
    save(fig, "training_curves")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.3), sharex=True)
    for ax, configuration in zip(axes.flat, learned_configurations):
        ax.axhline(0, color=".5", linewidth=.8)
        for override, color, offset in zip((0.0, .5, 1.0), palette[:3], (-.12, 0, .12)):
            rows = [next(row for row in interventions if (row["augmentation"], row["input"], row["condition"], row["gate_override"])
                         == (*configuration, condition, override)) for condition in CONDITIONS]
            ax.errorbar(x + offset, [row["macro_f1_delta_pp_mean"] for row in rows],
                        yerr=[row["macro_f1_delta_pp_sd"] for row in rows], fmt="o-", markersize=4,
                        capsize=3, color=color, linewidth=1.1, label=f"Same weights, g={override:g}")
        ax.set_title(LABELS[configuration])
        ax.set_ylabel("Change from learned gate (pp)")
        ax.set_xticks(x, [value.capitalize() for value in CONDITIONS])
        ax.grid(axis="y", alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False)
    fig.suptitle("Post-training gate interventions; mean +/- sample SD, n = 3\nSeparately trained fixed-mean models are excluded")
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, "intervention_effects")
    return names


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    columns = list(headers)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _relative(destination: Path, origin: Path) -> str:
    return Path(os.path.relpath(destination.resolve(), origin.resolve())).as_posix()


def build_report(study_dir: Path, report_dir: Path, runs: list[Run], summary: list[dict[str, Any]],
                 paired: list[dict[str, Any]], paired_summary: list[dict[str, Any]],
                 intervention_summary: list[dict[str, Any]], convergence: list[dict[str, Any]],
                 costs: list[dict[str, Any]], cost_summary: list[dict[str, Any]]) -> str:
    lookup = {(row["augmentation"], row["input"], row["condition"]): row for row in summary}
    score_rows = []
    for configuration in CONFIGURATIONS:
        values = []
        for condition in CONDITIONS:
            row = lookup[(*configuration, condition)]
            values.append(f"{100 * row['macro_f1_mean']:.2f} ± {100 * row['macro_f1_sd']:.2f}")
        score_rows.append([LABELS[configuration], *values])
    change_rows = [[row["comparison"], row["condition"], f"{row['macro_f1_delta_pp_mean']:+.2f} ± {row['macro_f1_delta_pp_sd']:.2f}",
                    str(row["positive_macro_f1_seeds"]) + "/3"] for row in paired_summary]
    all_pair_rows = [[row["comparison"], row["condition"], row["seed"], f"{row['macro_f1_delta_pp']:+.3f}"] for row in paired]
    learned_gates = [row for run in runs if run.config["input"] != "fixed" for row in run.gates
                     if row["split"] == "validation" and row["class_id"] == "all"]
    gate_rows = []
    for configuration in CONFIGURATIONS:
        if configuration[1] == "fixed":
            continue
        for condition in CONDITIONS:
            values = [row for row in learned_gates if (row["augmentation"], row["input"], row["condition"]) == (*configuration, condition)]
            mean, sd = mean_sd(row["gate_mean"] for row in values)
            gate_rows.append([LABELS[configuration], condition, f"{mean:.4f} ± {sd:.4f}",
                              f"{min(row['q05'] for row in values):.4f}–{max(row['q95'] for row in values):.4f}"])
    intervention_rows = [[LABELS[(row["augmentation"], row["input"])], row["condition"], f"{row['gate_override']:g}",
                          f"{100*row['macro_f1_mean']:.2f} ± {100*row['macro_f1_sd']:.2f}",
                          f"{row['macro_f1_delta_pp_mean']:+.2f} ± {row['macro_f1_delta_pp_sd']:.2f}"] for row in intervention_summary]
    cost_rows = [[LABELS[(row["augmentation"], row["input"])], f"{row['parameters_mean']:.0f}", row["threads"],
                 f"{row['training_seconds_mean']:.2f} ± {row['training_seconds_sd']:.2f}",
                 f"{row['export_seconds_mean']:.2f} ± {row['export_seconds_sd']:.2f}",
                 f"{row['clean_inference_ms_per_image_mean']:.4f} ± {row['clean_inference_ms_per_image_sd']:.4f}"] for row in cost_summary]
    convergence_rows = [[row["run_id"], f"{row['initial_train_loss']:.4f} → {row['final_train_loss']:.4f}",
                         f"{row['train_loss_change_last_epoch']:+.4f}", f"{row['validation_change_last_epoch_pp']:+.2f}",
                         row["best_validation_epoch_diagnostic_only"], f"{row['final_minus_best_validation_pp']:+.2f}"] for row in convergence]
    conclusions = []
    for name, candidate, reference in COMPARISONS:
        entries = [row for row in paired_summary if row["comparison"] == name]
        change_text = "；".join(f"{row['condition']} {row['macro_f1_delta_pp_mean']:+.2f} pp（{row['positive_macro_f1_seeds']}/3 种子为正）" for row in entries)
        conclusions.append(f"- **{LABELS[candidate]} 相对 {LABELS[reference]}**：{change_text}。")
    denoise = {row["condition"]: row for row in paired_summary if row["comparison"] == "denoised_minus_gated"}
    descriptive_target = (denoise["noise"]["macro_f1_delta_pp_mean"] >= 2 and
                          denoise["clean"]["macro_f1_delta_pp_mean"] >= -1 and
                          denoise["motion"]["macro_f1_delta_pp_mean"] >= -1 and
                          denoise["noise"]["positive_macro_f1_seeds"] >= 2)
    threads = sorted({int(run.cost["threads"]) for run in runs})
    warnings = "所有运行的线程数相同。" if len(threads) == 1 else "运行线程数不一致，耗时仅为记录值，不能据此作公平速度排序。"
    config_reference = _relative(runs[0].directory / "config.json", report_dir)
    study_link = _relative(study_dir, report_dir)
    total_train = sum(run.cost["training_seconds"] for run in runs)
    max_gap = max(abs(row["final_minus_best_validation_pp"]) for row in convergence)
    last_epoch_changes = [row["validation_change_last_epoch_pp"] for row in convergence]
    ordinary = [run for run in runs if run.config["input"] == "dual"]
    saturation_rows = []
    total_gate_one_changes = 0
    ordinary_gates = []
    for run in ordinary:
        gates = [row["gate"] for row in run.predictions]
        ordinary_gates.extend(gates)
        natural = {(row["condition"], row["image"]): row["y_pred"] for row in run.predictions}
        changes = sum(row["y_pred"] != natural[(row["condition"], row["image"])]
                      for row in run.intervention_predictions if row["gate_override"] == 1.0)
        total_gate_one_changes += changes
        saturation_rows.append([run.config["run_id"], f"{min(gates):.6f}",
                                f"{100*np.mean(np.asarray(gates) >= .99):.2f}%",
                                f"{100*np.mean(np.asarray(gates) == 1):.2f}%", changes])
    if min(ordinary_gates) >= .99 and total_gate_one_changes == 0:
        gate_interpretation = (
            f"普通 dual 的全部九次运行、五种条件共 {len(ordinary_gates):,} 条验证预测均满足 g≥0.99，"
            f"最小 g 为 {min(ordinary_gates):.6f}；同权重 g=1 干预改变的预测数为 0。"
            "因此，当前普通 dual 在这些开发图像的推理中近似选择 CLAHE 分支，"
            "没有展现按输入条件灵活融合两路信息的证据。g=1 时分类结果保持一致进一步支持这一描述。"
            "这不等同于独立训练的 CLAHE 单分支：编码器和分类头是在双分支训练过程中共同优化的，仍需 A/B 正式对照。")
    else:
        gate_interpretation = (
            f"普通 dual 的验证门控最小值为 {min(ordinary_gates):.6f}，"
            f"g=1 干预共改变 {total_gate_one_changes} 条预测。应结合分布及下表逐运行值解释权重偏好，"
            "不能仅凭均值较高声称门控有输入自适应能力。")
    paired_lookup = {(row["comparison"], row["condition"]): row for row in paired_summary}
    trade_noise = paired_lookup[("mixed_minus_targeted", "noise")]
    trade_motion = paired_lookup[("mixed_minus_targeted", "motion")]
    denoise_noise = paired_lookup[("denoised_minus_gated", "noise")]
    lead = (f"核心结果：mixed 门控相对 targeted_motion 门控的噪声 Macro-F1 平均变化为 "
            f"{trade_noise['macro_f1_delta_pp_mean']:+.2f} pp，运动模糊为 {trade_motion['macro_f1_delta_pp_mean']:+.2f} pp。"
            f"去噪门控相对 mixed 门控的噪声差值为 {denoise_noise['macro_f1_delta_pp_mean']:+.2f} pp；"
            f"预设描述性噪声收益检查{'满足' if descriptive_target else '未全部满足'}。"
            + ("普通 dual 的全部门控 g≥0.99 且 g=1 干预不改变预测，支持其在本次验证中近似选择 CLAHE 分支的描述。"
               if min(ordinary_gates) >= .99 and total_gate_one_changes == 0
               else "门控偏好及预测敏感性分别由实际分布和同权重干预结果描述。"))
    checkpoint_note = "本报告生成时未发现已通过的独立权重重载核验记录；指纹完整性与 NPZ/CSV 对齐检查不代替重载执行。"
    checkpoint_path = study_dir / "checkpoint_verification.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (checkpoint.get("status") == "passed" and checkpoint.get("runs") == 15
                and checkpoint.get("data_fingerprint") == runs[0].config["data_fingerprint"]
                and all(checkpoint.get(key) == runs[0].config[key] for key in
                        ("backbone", "feature_dim", "network_input_size", "weights_id", "weights_sha256"))):
            checkpoint_note = (f"独立重载核验记录 [checkpoint_verification.json]({study_link}/checkpoint_verification.json) 已通过："
                               f"15 次运行共 {checkpoint['feature_prediction_gate_images']:,} 个图像/条件实例的特征、预测与门控，"
                               f"以及 {checkpoint['intervention_images']:,} 条干预预测与原始导出匹配。"
                               "这项记录由训练后的独立核验产生，与本汇总程序的指标重算分别执行。")
    smoke_note = ("本次十五次矩阵没有 raw 单分支计时对照，因此无法从正式表量化融合相对单分支的额外前向成本。"
                  "工程小样本核验与历史 SmallCNN 计时不混入本研究。应在同一机器、线程、视图和计时方法下补做重复测量。")
    return f"""# C 部分：官方预训练 ResNet18 与既有融合消融报告

状态：已生成并核验的开发验证报告。读者为项目小组及课程报告整合者。数据来自 `{study_dir.as_posix()}` 的 15 次训练，不包含官方测试。

本次对五种配置各运行三个种子，汇报最终第 10 轮的七类分类结果、配对差值、门控干预和成本。下述差值是固定开发划分上的描述结果，不能据此宣称已经通过独立测试或证明门控有因果机制。

{lead}

## 1. 范围与可追溯证据

- 运行矩阵：门控双分支 × repeat_clean / targeted_motion / mixed × 42、43、44，共 9 次；mixed 下 fixed 与 dual_denoised 各三个种子，共 6 次。
- 类别顺序为 1、2、3、4、5、7、8。固定训练集 2,940 帧、98 条轨迹，开发验证集 840 帧、28 条轨迹，每条轨迹 30 帧，训练和验证没有轨迹交叉。
- 每次训练含 4,410 个条目，其中 2,940 张干净图和 1,470 个额外条目。repeat_clean 的额外条目是原图重复，不是新增独立图像；mixed 的三类干扰各 490 个条目。
- 本程序核验 15 个配置和完整性记录，独立复算 75 条常规指标及 180 条干预指标；共有 63,000 条常规验证预测、151,200 条干预预测和 90 个特征 NPZ。验证条件共用同一批 840 张图，不能将五种条件当成 4,200 个独立样本。
- 原始证据：[运行目录]({study_link}/runs/)、[配置示例]({config_reference})、[聚合指标]({study_link}/metrics.csv)、[逐图预测]({study_link}/predictions.csv)、[核验清单](summary_manifest.json)。完整性指纹验证不等同于重新训练得到逐位一致的结果。
- 本次未新增正式的 raw、CLAHE 或 denoised-CLAHE 单分支对照。历史 SmallCNN 结果与旧预实验分数均不混入当前 ResNet 表；骨干、ImageNet 初始化及输入尺寸同时变化，不能把两套结果解释为只隔离骨干结构的对照。A/B 后续的单分支与统一官方测试仍为缺失证据。

## 2. 模型、训练与视图

共享骨干为 torchvision 官方 `resnet18`，以 `ResNet18_Weights.IMAGENET1K_V1` 的 ImageNet 权重初始化。保留官方卷积 stem、BatchNorm、残差层、下采样和平均池化，直接取平均池化后的 512 维表示，七类分类头替换官方 1,000 类头；不添加自定义残差块、stem、注意力或投影层。原图与增强图共享同一个编码器，并完整微调骨干与分类头。既有双视图融合和消融沿用项目方案，不作为新发明。学习门控由两路特征拼接经线性层和 sigmoid 生成标量 g，融合为 `(1-g)·z_raw + g·z_enhanced`。g 越大，增强视图权重越大。

缓存视图、合成干扰及 CLAHE 均先在既有 32×32 图上构建，再对每一批视图应用官方权重预处理 `Resize(256) → CenterCrop(224) → ImageNet Normalize`。配置的 `image_size=32` 指源视图尺寸，`network_input_size=224` 才是骨干输入；单分支与融合使用相同预处理，不加入随机裁剪或翻转。

fixed 模型独立训练，在相同官方 ImageNet 初始化后完整微调，并在所有样本上固定 g=0.5。dual_denoised 保留原图分支，将增强分支改为双边滤波后 CLAHE：`d=5, sigmaColor=25, sigmaSpace=2`。CLAHE 仅作用于 LAB 的 L 通道，`clipLimit=2.0, tileGridSize=(4,4)`。干扰先作用于 8 位受损图，再生成两种配对视图；增强分支不接收对应干净图。

统一使用 AdamW、学习率 0.003、weight decay 0.01、betas=(0.9, 0.999)、epsilon=1e-8、batch size 128 和 10 轮；每轮 35 次优化更新，共 350 步。三个种子为 42、43、44，所有结果都使用最终轮，不用各组最佳轮挑选成绩。clean、motion、noise、dark、jpeg 是当前五个开发条件；`severity` 的实际标记保留在 CSV，不能将这里的五条件结果写成 13 条件或多强度结论。JPEG 没有加入训练增强。

## 3. 最终轮开发验证结果

表内为 Macro-F1 百分比的三种子均值 ± 样本标准差（`ddof=1`）。准确率的相同统计见 [完整汇总](tables/metric_summary.csv)。标准差描述训练随机性；n=3 很小，不能解释为精确的总体误差或置信区间。

{markdown_table(("配置", *CONDITIONS), score_rows)}

![Final-epoch development scores](figures/scores_with_sd.png)

## 4. 按种子配对的比较

每个差值使用同种子、同划分和同条件的两个模型，单位为百分点（pp）。表内的差值标准差直接由三条配对差值计算，不能由两组标准差相减替代。

{markdown_table(("比较（候选减参照）", "条件", "Macro-F1 差值均值 ± SD (pp)", "正差值种子"), change_rows)}

{chr(10).join(conclusions)}

这些比较完整保留收益和退步，属于描述性证据；当前没有进行显著性检验。去噪融合相对原门控在噪声至少 +2 pp、clean/motion 不低于 −1 pp、噪声至少 2/3 种子为正这一描述性检查：**{'满足' if descriptive_target else '未全部满足'}**。这只对 C 的 mixed 门控参照成立；原计划的主要参照是 CLAHE 单分支，该正式对照尚未生成，因此不能据此宣称完整通过项目的实践价值规则。

![Paired changes](figures/paired_changes.png)

## 5. 门控分布与解释边界

下表先在每个运行的 840 帧上求门控均值，再对三个种子均值计算均值 ± SD。“各运行 5–95% 包络”是三个运行中 q05 的最小值至 q95 的最大值，用于观察分布，不是置信区间。逐运行的帧级均值、SD、最小值、5/25/50/75/95% 分位数、最大值及每类分布见 [gate_distribution.csv](tables/gate_distribution.csv)。固定平均模型的 g 恒为 0.5。

{gate_interpretation}

下表每行合并同一运行的五个验证条件（4,200 条预测），分别列出近饱和和数值上恰好为 1 的比例；这些重复图像实例用于描述，不是独立样本。

{markdown_table(("普通 dual 运行", "最小 g", "g≥0.99 比例", "g=1 比例", "强制 g=1 改变预测数"), saturation_rows)}

{markdown_table(("配置", "条件", "运行门控均值 ± 种子 SD", "各运行 5–95% 包络"), gate_rows)}

![Gate distributions](figures/gate_distribution.png)

g 随条件或类别变化只说明当前模型的权重分配与这些分组相关，不能证明门控“识别了噪声”、不能把 g 当成解释性交通语义，也不能由分布变化推出性能收益的因果机制。图中为重复使用的帧分布，帧与同轨迹图像相关；没有把这些帧当成独立重复来生成推断区间。

## 6. 同一权重下的门控干预

本节仅在已经训练好的 learned 模型上替换 g，保持编码器和分类头权重不变。g=0 使用原图特征，g=1 使用增强视图特征，g=0.5 使用两者平均。端点分支仍是在融合训练过程中共同学习得到，不能作为独立训练的单分支模型。这里的 g=0.5 干预与第 3 节独立训练的 fixed 模型是两种不同实验。

{markdown_table(("训练配置", "条件", "干预 g", "Macro-F1 (%) 均值 ± SD", "相对学习门控 Δ (pp) 均值 ± SD"), intervention_rows)}

![Post-training gate interventions](figures/intervention_effects.png)

干预表反映改变当前融合规则时的预测敏感性，不能单独证明门控的学习机制在真实场景中具有因果优势。完整逐种子结果见 [intervention_by_run.csv](tables/intervention_by_run.csv)。

## 7. 训练状态与固定轮数的限制

![Training curves](figures/training_curves.png)

表内“最佳轮”仅作为训练状态诊断，不参与模型选择。15 次运行末轮相对各自最佳 clean 验证分数的最大落差为 {max_gap:.2f} pp，最后一轮 clean 分数变化范围为 {min(last_epoch_changes):+.2f} 至 {max(last_epoch_changes):+.2f} pp。有限的 10 轮轨迹不能证明全部模型充分收敛；若末端仍变化，应在后续统一协议中检验更长训练，而不能只延长某个落后配置。

{markdown_table(("运行", "首轮 → 末轮训练损失", "末轮损失变化", "末轮验证变化 (pp)", "诊断最佳轮", "末轮减最佳 (pp)"), convergence_rows)}

## 8. 参数与时间成本

表内为三种子均值 ± 样本 SD。训练共计 {total_train:.2f} 秒（累加各次记录，不代表并行任务的实际总墙钟时间）；训练计时排除逐轮验证和导出。{warnings} 前向记录包括已准备的 32×32 视图的批次选择、设备传输、模型内部的官方 Resize/CenterCrop/Normalize、ResNet 前向及 CPU 输出收集，按同一批验证图摊算；它不包括源 PPM 磁盘读取或 32×32 干扰/CLAHE 视图缓存构建，且不是逐张在线延迟的测量。训练、导出和各评估条件前向耗时分别见 [逐运行成本](tables/cost_by_run.csv) 与 [条件耗时](tables/inference_cost.csv)。`preprocessing_seconds` 记录的是 CPU 源视图构建，每次同时生成 raw、CLAHE 和 denoised-CLAHE 三种视图，可被各运行复用，不应再次加总为独立执行成本，也不能拆解为某种单视图方法的耗时。

{markdown_table(("配置", "参数", "线程", "训练 (s)", "导出 (s)", "clean 前向 (ms/图)"), cost_rows)}

这些成本只对应记录中的环境及当前执行方式。预处理对去噪方案的代价需要与前向耗时分开看。

{smoke_note}

## 9. 核验、局限与后续交接

汇总程序先执行 `verify_completed_run(run_dir, config)` 检查每个运行，再独立检查矩阵、固定协议、七类标签、75 条常规指标、180 条干预指标、预测重复、图像/轨迹/标签配对、六个 NPZ 的 512 维特征及元数据，以及末轮训练日志与导出指标一致性。Macro-F1 始终固定包含全部七类，零分母类别按 0 计；准确率从逐图预测重算。完整 source 文件 SHA-256 及生成时间保留于 [summary_manifest.json](summary_manifest.json)。

官方权重来源为 `ResNet18_Weights.IMAGENET1K_V1`，完整 SHA-256 为 `{runs[0].config["weights_sha256"]}`，记录的 torchvision 为 `{runs[0].config["torchvision_version"]}`。汇总前核对项目内权重文件的完整校验和；不会下载或替换权重。

研究根部 `run_config.json` 的矩阵必须与十五种组合完全一致；研究内 `split.csv` 的完整字节 SHA-256 必须等于公共配置，`data_manifest.json` 的字段与训练入口一致，并从源文件记录、原始划分顺序及三种训练/五种验证像素指纹独立重算稳定数据指纹。汇总不重新生成像素，也不要求当前训练代码哈希与历史快照相同。每次运行去掉运行 ID、种子、输入、增强和预处理计时字段后，其余所有配置字段必须与研究公共配置逐项相等，缺失或额外字段都会拒绝汇总。

{checkpoint_note}

当前报告限定于七类、一个固定开发划分、28 条验证轨迹、合成干扰、五个条件和三个训练种子。不同帧、条件、种子都复用同一验证图，不能把 63,000 行当成 63,000 个独立观测。本文没有构建逐帧 IID 置信区间；D 如需配对 bootstrap，应以整条轨迹为单位、按类别分层，在一次重采样中同时保留所有条件与配对种子，并明确区间仅描述这一固定开发划分。

后续由 A/B 补齐正式单分支对照与统一多强度评估，由 D 在干净训练特征上拟合 PCA/KMeans 并验证开发表示，不在验证特征上拟合后报告泛化。全组先冻结候选清单再打开官方七类测试集。官方测试未在本报告中使用，也不能把本报告趋势外推到全部 43 类或真实道路。

复现命令、产物契约、失败处理及环境信息入口见 [README](README.md)。

## 附录：全部逐种子配对差值

{markdown_table(("比较", "条件", "种子", "Macro-F1 Δ (pp)"), all_pair_rows)}
"""


def build_readme(study_dir: Path, report_dir: Path, runs: list[Run], figure_names: list[str]) -> str:
    study_path = _relative(study_dir, PROJECT_ROOT)
    report_path = _relative(report_dir, PROJECT_ROOT)
    environments = sorted({json.dumps(run.config.get("environment", {}), sort_keys=True, ensure_ascii=False) for run in runs})
    reproduction_path = f"{study_path}_reproduction"
    recorded_threads = sorted({int(run.cost["threads"]) for run in runs})
    recorded_devices = sorted({str(run.config.get("device", "cpu")) for run in runs})
    weights_path = runs[0].config["weights_path"]
    return f"""# 官方预训练 ResNet18：C 部分结果汇总与复现

状态：完整十五次运行已通过本汇总程序的配置、完整性、特征对齐和指标重算检查。本目录只覆盖固定开发验证；不代表官方测试通过，也不代表实体 Spark 上可逐位重现。此 README 仅在完整矩阵核验后生成；尚未完成正式训练时，本程序不会生成结果报告。

面向接手 C 产物的小组成员。模型使用 torchvision 官方 `resnet18`、`ResNet18_Weights.IMAGENET1K_V1`，共享官方骨干，保留既有双视图融合；不将融合写成新发明。导出特征为 512 维。先在 32×32 源视图上生成干扰与 CLAHE，再应用官方 `Resize(256) → CenterCrop(224) → ImageNet Normalize`。骨干、预训练与网络输入尺寸同时改变，历史 SmallCNN 结果单独保留，不能当作隔离架构因素的比较。

## Spark 默认入口

在解压后的项目根目录、已准备好的 Spark 容器运行环境使用：

```bash
bash SPARK.sh train
bash SPARK.sh verify
bash SPARK.sh report
```

默认正式研究为 `results/study_resnet18_spark_v1`，报告为 `report/c_part_resnet18_spark`。`train` 执行十五次矩阵，`verify` 核验训练产物，`report` 独立重算并生成报告。已有完整研究时只执行后两条；中断恢复使用 `bash SPARK.sh train --resume`。具体容器、匹配的 Torch/Torchvision、GPU 和首次启动条件见项目 Spark 说明；不要向 NVIDIA 容器安装覆盖其原有 Torch/Torchvision 的 CPU 包。

自定义目录时，在同一已准备的容器内使用以下直接入口。重训目录必须尚不存在，保留本次结果：

```bash
python tools/run_spark_study.py --output-dir "{reproduction_path}" --epochs 10 --threads {recorded_threads[0]} --device {recorded_devices[0]} --weights-path "{weights_path}"
python experiments/run_resnet_study.py --output-dir "{study_path}" --verify-only
python experiments/summarize_resnet_study.py --study-dir "{study_path}" --report-dir "{report_path}"
```

第一条将结果写入新的复现目录，第二、三条校验和汇总当前报告的原始研究；要汇总新复现结果，应同时将第二、三条的研究目录改为 `{reproduction_path}`，并指定独立报告目录。训练中断可在相同直接训练命令添加 `--resume`。恢复只接受冻结配置、代码、环境及完成指纹一致的运行，半成品会停止，不能手改 CSV 或删除指纹绕过检查。

本次设备记录为 `{','.join(recorded_devices)}`，线程记录为 `{','.join(map(str, recorded_threads))}`。若存在多个值，应分别按各运行配置复现；更改硬件、线程或依赖可能改变时间和浮点结果。源配置记录的环境如下；版本缺失时不从汇总机器推断训练环境：

```json
{json.dumps([json.loads(value) for value in environments], indent=2, ensure_ascii=False)}
```

官方初始化文件为 `{weights_path}`，完整 SHA-256 为 `{runs[0].config['weights_sha256']}`；torchvision 版本为 `{runs[0].config['torchvision_version']}`。必须保留项目内相对路径和现有权重文件，汇总会重新校验完整文件，不隐式下载。汇总入口仅读取配置和产物，调用通用完成记录核验；不构建 torchvision 模型或执行前向。NumPy、Matplotlib 与通用核验模块依赖需在运行环境中可用。

## 输入、输出与验收

输入矩阵为 dual × repeat_clean/targeted_motion/mixed，加 mixed × fixed/dual_denoised，各种子 42、43、44。必须恰好十五次运行，配置的全部公共字段与研究 `run_config.json` 完全一致，包含官方骨干、预训练、权重校验和、512 维与 224 输入元数据。

| 每运行产物 | 验收 |
| --- | --- |
| config.json、model.pt、完成指纹 | 官方 ResNet18 身份与冻结配置、源文件完整性 |
| metrics.csv、predictions.csv | 五条件、4,200 条预测，同一批 840 张、28 条轨迹；固定七类重算 Macro-F1/准确率 |
| training_log.csv | 十轮，末轮 clean 指标与导出一致 |
| features/*.npz | 六份：2,940 张 train_clean 及五份 840 张验证集；512 维，图像/轨迹/真值/预测/gate 对齐 |
| intervention_metrics.csv、intervention_predictions.csv | learned 模型 15 条指标、12,600 条预测，g=0、0.5、1；fixed 无此产物 |
| cost.json | 实际参数、训练/导出/前向耗时、4,410 条目、350 步及线程 |

前向成本包含批次选择、设备复制、模型内部官方 Resize/CenterCrop/Normalize、ResNet 计算与 CPU 输出收集；只排除源 PPM 读取及 32×32 干扰/CLAHE 视图缓存构建。配置中的 `preprocessing_seconds` 是可跨运行复用的 CPU 视图构建记录，不将它按每个模型重复累加。

研究目录必须保留自身的 `split.csv` 与 `data_manifest.json`；不以外部划分文件替代缺失快照。汇总核对 split 的完整字节 SHA-256，并从清单中记录的源文件哈希、原始 split 顺序和三种训练/五种验证像素哈希重算稳定数据指纹，不读取原始图像或重新生成干扰。成功应显示 15 次运行、75 条常规指标与 63,000 条常规预测，另外核验 180 条干预指标、151,200 条干预预测和 90 份 NPZ。三种子样本标准差使用 `ddof=1`，配对差值逐种子计算；帧级门控统计不作 IID 推断。

程序仅在全部来源通过检查后重建派生文件：

- 研究根部 `metrics.csv`、`predictions.csv` 及两份 intervention 总表。
- `tables/`：指标均值和 SD、逐种子配对差值、门控分布、同权重干预、训练轨迹和实际成本。
- `figures/`：{', '.join(figure_names)}，各有 PNG 与 SVG。
- [C_REPORT.md](C_REPORT.md)：模型、实测结果和证据边界；[summary_manifest.json](summary_manifest.json)：源文件 SHA-256、官方权重与可用 Spark 环境记录、核验计数及输出清单。

报告生成会重建派生文件，每次原始运行只读。来源缺失、历史骨干、预训练身份不符、非 512 维或预测不对齐时，以非零状态停止，不生成部分矩阵报告。绘图或文件写入失败后，保留所有输入并修复运行环境，再用相同汇总命令重建；以最后写出的 `summary_manifest.json` 作为完整生成标志。

## 失败处理与边界

数量/公共配置不符时，C 应检查原始运行与冻结清单。指纹、权重校验和或指标重算失败时停止使用对应产物，在独立输出目录重新训练，不手改来源文件。门控或元数据不一致时检查导出排序、原始类别编号及轨迹标识，不能重新排列标签掩盖错误。

本研究未提供正式单分支对照、多强度评估、独立测试或总体置信区间；同权重端点干预不等于独立训练分支。A/B 补齐正式对照，D 如需 bootstrap 应按整条轨迹配对重采样。协议依据为 `03-resnet-c-implementation.md` 及其保留的 C 矩阵；历史 SmallCNN 成果不合并进本表。
"""


def source_manifest(runs: list[Run], study_dir: Path) -> dict[str, Any]:
    sources = []
    source_paths = []
    for run in runs:
        source_paths.extend(path for path in sorted(run.directory.rglob("*")) if path.is_file())
    required_study_sources = [study_dir / name for name in ("run_config.json", "split.csv", "data_manifest.json")]
    for path in required_study_sources:
        require(path.is_file(), f"Missing required study source: {path}")
    source_paths.extend(required_study_sources)
    source_paths.extend(path for path in (study_dir / "checkpoint_verification.json",
                                          study_dir / "spark_runtime.json", Path(__file__).resolve(),
                                          Path(__file__).with_name("cnn_robustness_study.py"),
                                          _verified_weights_path(runs[0].config))
                        if path.is_file())
    for path in dict.fromkeys(source_paths):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        sources.append({"path": _relative(path, study_dir), "bytes": path.stat().st_size,
                        "sha256": digest.hexdigest()})
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "development_validation_only",
            "runs": len(runs), "training_seeds": list(SEEDS), "classes": list(CLASSES),
            "backbone": BACKBONE, "pretrained": True, "feature_dim": FEATURE_DIM,
            "network_input_size": NETWORK_INPUT_SIZE, "source_image_size": 32,
            "weights_id": WEIGHTS_ID, "weights_sha256": WEIGHTS_SHA256,
            "weights_path": runs[0].config["weights_path"],
            "torchvision_version": runs[0].config["torchvision_version"],
            "metric_rows": sum(len(run.metrics) for run in runs),
            "validation_prediction_rows": sum(len(run.predictions) for run in runs),
            "intervention_metric_rows": sum(len(run.interventions) for run in runs),
            "intervention_prediction_rows": sum(len(run.intervention_predictions) for run in runs),
            "feature_archives": sum(len(list((run.directory / "features").glob("*.npz"))) for run in runs), "training_images": 2940, "validation_images": 840,
            "independent_training_tracks": 98, "independent_validation_tracks": 28,
            "standard_deviation_ddof": 1, "confidence_intervals": "not_computed",
            "checks": ["official_resnet18_pretraining_identity", "official_weights_file_sha256", "study_data_manifest_snapshot_fingerprint", "study_split_file_sha256", "trainer_completed_run_integrity", "exact_frozen_config_matrix", "frozen_common_configuration_identity", "fixed_split_and_track_isolation",
                       "fixed_class_metric_recomputation_including_interventions", "feature_prediction_metadata_alignment",
                       "final_epoch_export_consistency", "training_budget_and_cost_fields"],
            "data_fingerprint": runs[0].config["data_fingerprint"], "sources": sources}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate all 15 official pretrained ResNet18 C-study runs and generate development-validation tables, figures and Chinese report.")
    parser.add_argument("--study-dir", type=Path, default=Path("results/study_resnet18_spark_v1"))
    parser.add_argument("--report-dir", type=Path, default=Path("report/c_part_resnet18_spark"))
    arguments = parser.parse_args(argv)
    study_dir, report_dir = arguments.study_dir.resolve(), arguments.report_dir.resolve()
    require(report_dir != study_dir and not report_dir.is_relative_to(study_dir / "runs"),
            "Report directory must not overlap source run artifacts")
    runs = load_study(study_dir)
    metrics = [row for run in runs for row in run.metrics]
    predictions = [row for run in runs for row in run.predictions]
    intervention_metrics = [row for run in runs for row in run.interventions]
    intervention_predictions = [row for run in runs for row in run.intervention_predictions]
    require((len(metrics), len(predictions), len(intervention_metrics), len(intervention_predictions)) == (75, 63000, 180, 151200),
            "Unexpected aggregate size")
    summary = summarize_metrics(metrics)
    paired, paired_summary = paired_differences(metrics)
    intervention_by_run, intervention_summary = summarize_interventions(runs)
    training, convergence, costs, inference = convergence_and_cost(runs)
    cost_summary = summarize_costs(costs)
    manifest = source_manifest(runs, study_dir)
    # No source or aggregate output is changed until the complete matrix validates.
    write_csv(study_dir / "metrics.csv", metrics, METRIC_FIELDS)
    write_csv(study_dir / "predictions.csv", predictions, PREDICTION_FIELDS)
    write_csv(study_dir / "intervention_metrics.csv", intervention_metrics, (*METRIC_FIELDS, "gate_override"))
    write_csv(study_dir / "intervention_predictions.csv", intervention_predictions, (*PREDICTION_FIELDS, "gate_override"))
    tables = {"metric_summary": summary, "paired_by_seed": paired, "paired_summary": paired_summary,
              "intervention_by_run": intervention_by_run, "intervention_summary": intervention_summary,
              "gate_distribution": [row for run in runs for row in run.gates], "training_log": training,
              "convergence": convergence, "cost_by_run": costs, "cost_summary": cost_summary,
              "inference_cost": inference}
    for name, rows in tables.items():
        write_csv(report_dir / "tables" / f"{name}.csv", rows)
    figure_names = make_figures(report_dir, runs, summary, paired, paired_summary, intervention_summary)
    write_text(report_dir / "C_REPORT.md", build_report(study_dir, report_dir, runs, summary, paired, paired_summary,
                                                       intervention_summary, convergence, costs, cost_summary))
    write_text(report_dir / "README.md", build_readme(study_dir, report_dir, runs, figure_names))
    manifest["outputs"] = [str(path.relative_to(report_dir).as_posix()) for path in sorted(report_dir.rglob("*"))
                           if path.is_file() and path.name != "summary_manifest.json"]
    write_text(report_dir / "summary_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Validated 15 runs, 75 metric rows, 63000 validation predictions; report: {report_dir / 'C_REPORT.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StudyValidationError, FileNotFoundError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"ResNet18 C study summary failed: {error}", file=sys.stderr)
        raise SystemExit(1)
