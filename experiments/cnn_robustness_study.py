"""
用途：提供七类交通标志的公共训练、融合干预与可追溯结果导出。
核心职责：
- 统一原图、CLAHE、去噪视图和共享编码器的训练行为。
- 保存最终模型、逐轮日志、对齐特征、预测与门控干预。
- 校验已完成运行的配置和产物指纹。
关键副作用/安全约束：
- 仅在新的运行目录写入结果，拒绝覆盖已有运行。
- 不读取官方测试集，不按验证分数挑选训练轮次。
"""

import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


CLASSES = (1, 2, 3, 4, 5, 7, 8)
INPUT_MODES = {
    "raw": ("single", ("raw",)),
    "clahe": ("single", ("clahe",)),
    "denoised_clahe": ("single", ("denoised_clahe",)),
    "dual": ("learned", ("raw", "clahe")),
    "fixed": ("fixed", ("raw", "clahe")),
    "dual_denoised": ("learned", ("raw", "denoised_clahe")),
    "dual_spatial": ("spatial", ("raw", "denoised_clahe")),
}
METRIC_FIELDS = (
    "run_id", "seed", "augmentation", "input", "split", "condition", "severity",
    "macro_f1", "accuracy", "n_images", "parameters",
)
PREDICTION_FIELDS = (
    "run_id", "seed", "split", "image", "track_id", "condition", "severity",
    "y_true", "y_pred", "gate",
)


class SmallCNN(nn.Module):
    """The pilot encoder with single, learned-gate, or trained-average fusion."""

    def __init__(self, fusion="single"):
        super().__init__()
        if fusion not in ("single", "learned", "fixed"):
            raise ValueError(f"Unknown fusion mode: {fusion}")
        self.fusion = fusion
        layers = []
        for channels_in, channels_out in ((3, 12), (12, 24), (24, 32)):
            layers.extend((
                nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
                nn.GroupNorm(4, channels_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
        self.encoder = nn.Sequential(
            *layers, nn.Flatten(), nn.Linear(32 * 4 * 4, 32), nn.ReLU()
        )
        self.classifier = nn.Linear(32, len(CLASSES))
        self.gate = nn.Linear(64, 1) if fusion == "learned" else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward_details(self, image, enhanced=None, gate_override=None):
        if self.fusion == "single":
            if enhanced is not None or gate_override is not None:
                raise ValueError("A single branch accepts one view and no gate override")
            features = self.encoder(image)
            return {
                "logits": self.classifier(features),
                "features": features,
                "raw_features": features,
                "gate": features.new_full((len(features), 1), float("nan")),
            }
        if enhanced is None or image.shape != enhanced.shape:
            raise ValueError("Fusion requires paired views of equal shape")
        raw_features, enhanced_features = self.encoder(torch.cat((image, enhanced))).chunk(2)
        if gate_override is not None:
            value = float(gate_override)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Gate override must be finite and within [0, 1]")
            weight = raw_features.new_full((len(raw_features), 1), value)
        elif self.gate is None:
            weight = raw_features.new_full((len(raw_features), 1), 0.5)
        else:
            weight = self.gate(torch.cat((raw_features, enhanced_features), dim=1)).sigmoid()
        features = torch.lerp(raw_features, enhanced_features, weight)
        return {
            "logits": self.classifier(features),
            "features": features,
            "raw_features": raw_features,
            "enhanced_features": enhanced_features,
            "gate": weight,
        }

    def forward(self, image, enhanced=None, gate_override=None):
        return self.forward_details(image, enhanced, gate_override)["logits"]


def tensor_views(images):
    """Build paired RGB tensors from already resized, possibly corrupted BGR images."""
    import cv2

    if len(images) == 0:
        raise ValueError("At least one image is required")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    collected = {name: [] for name in ("raw", "clahe", "denoised_clahe")}
    for source in images:
        image = np.asarray(source)
        if image.dtype != np.uint8 or image.shape != (32, 32, 3):
            raise ValueError("Views require 32x32 uint8 BGR images")
        collected["raw"].append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for name, view in (
            ("clahe", image),
            ("denoised_clahe", cv2.bilateralFilter(image, 5, 25, 2)),
        ):
            lab = cv2.cvtColor(view, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            collected[name].append(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    return {
        name: torch.from_numpy(np.stack(view)).permute(0, 3, 1, 2).float().div_(255)
        .contiguous(memory_format=torch.channels_last)
        for name, view in collected.items()
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config):
    encoded = json.dumps(
        config, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _select_views(views, mode, indices, device="cpu"):
    return tuple(
        views[name][indices].to(device=device, memory_format=torch.channels_last)
        for name in INPUT_MODES[mode][1]
    )


def _validate_views(views, mode, n_images):
    for name in INPUT_MODES[mode][1]:
        if name not in views:
            raise ValueError(f"Missing input view: {name}")
        value = views[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (n_images, 3, 32, 32):
            raise ValueError(f"Invalid tensor shape for {name}")
        if value.device.type != "cpu" or not value.is_floating_point():
            raise ValueError("Stage datasets as floating-point CPU views; batches move to the training device")
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise ValueError(f"View {name} contains non-finite or out-of-range pixels")


def _scores(y_true, y_pred, classes):
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate(model, mode, spec, classes, batch_size, gate_override=None):
    rows = spec["rows"]
    order = sorted(range(len(rows)), key=lambda index: rows[index]["image"])
    collected = {"features": [], "gate": [], "raw_features": [], "y_pred": []}
    if model.fusion != "single":
        collected["enhanced_features"] = []
    model.eval()
    device = next(model.parameters()).device
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = torch.tensor(order[start:start + batch_size], dtype=torch.long)
            details = model.forward_details(
                *_select_views(spec["views"], mode, indices, device), gate_override=gate_override
            )
            for key in collected:
                if key == "y_pred":
                    collected[key].append(np.asarray(classes)[details["logits"].argmax(1).cpu().numpy()])
                else:
                    collected[key].append(details[key].cpu().numpy())
    _synchronize(device)
    elapsed = time.perf_counter() - started
    output = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    output.update({
        "image": np.asarray([rows[index]["image"] for index in order], dtype=str),
        "track_id": np.asarray([rows[index].get("track_id", "") for index in order], dtype=str),
        "y_true": np.asarray([rows[index]["class_id"] for index in order], dtype=np.int64),
    })
    if not np.isfinite(output["features"]).all():
        raise ValueError("Model produced non-finite features")
    if model.fusion != "single" and not np.isfinite(output["gate"]).all():
        raise ValueError("Model produced non-finite gates")
    return output, elapsed


def _metric_row(config, spec, output, parameters):
    return {
        "run_id": config["run_id"], "seed": config["seed"],
        "augmentation": config["augmentation"], "input": config["input"],
        "split": spec["split"], "condition": spec["condition"],
        "severity": spec.get("severity", "none" if spec["condition"] == "clean" else "medium"),
        **_scores(output["y_true"], output["y_pred"], config["classes"]),
        "n_images": len(output["image"]), "parameters": parameters,
    }


def _prediction_rows(config, spec, output, gate_override=None):
    severity = spec.get("severity", "none" if spec["condition"] == "clean" else "medium")
    for index, image in enumerate(output["image"]):
        gate = float(output["gate"][index, 0])
        row = {
            "run_id": config["run_id"], "seed": config["seed"],
            "split": spec["split"], "image": image,
            "track_id": output["track_id"][index], "condition": spec["condition"],
            "severity": severity, "y_true": int(output["y_true"][index]),
            "y_pred": int(output["y_pred"][index]), "gate": "" if math.isnan(gate) else gate,
        }
        if gate_override is not None:
            row["gate_override"] = gate_override
        yield row


def fit_run(run_dir, config, train_views, train_targets, evaluations):
    """Train the final epoch and write a fingerprinted, independently reloadable run."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    config = json.loads(json.dumps(config, allow_nan=False))
    mode = config["input"]
    if mode not in INPUT_MODES:
        raise ValueError(f"Unknown input mode: {mode}")
    classes = config.get("classes", list(CLASSES))
    if classes != list(CLASSES):
        raise ValueError("The seven-class order must remain [1,2,3,4,5,7,8]")
    config["classes"] = classes
    epochs, batch_size = int(config["epochs"]), int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    threads = int(config.get("threads", torch.get_num_threads()))
    if threads < 1:
        raise ValueError("Thread count must be positive")
    torch.set_num_threads(threads)
    device = torch.device(config.get("device", "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Training device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    train_targets = torch.as_tensor(train_targets, dtype=torch.long, device="cpu")
    if train_targets.ndim != 1 or len(train_targets) == 0:
        raise ValueError("Training targets must be a nonempty vector")
    if int(train_targets.min()) < 0 or int(train_targets.max()) >= len(classes):
        raise ValueError("Training targets must use contiguous class indices 0 to 6")
    _validate_views(train_views, mode, len(train_targets))
    if not evaluations:
        raise ValueError("At least one evaluation is required")
    for name, spec in evaluations.items():
        if not name or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Evaluation names must be simple filenames")
        rows = spec["rows"]
        if not rows or len({row["image"] for row in rows}) != len(rows):
            raise ValueError("Evaluation images must be nonempty and unique")
        if any(row["class_id"] not in classes or row["split"] != spec["split"] for row in rows):
            raise ValueError("Evaluation labels or split metadata are inconsistent")
        _validate_views(spec["views"], mode, len(rows))
    clean_validation = next((
        spec for spec in evaluations.values()
        if spec["split"] == "validation" and spec["condition"] == "clean"
    ), None)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "status.json", {"status": "running", "run_id": config["run_id"]})
    try:
        torch.manual_seed(int(config["seed"]))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(config["seed"]))
        batches = torch.Generator().manual_seed(int(config["seed"]))
        model = SmallCNN(INPUT_MODES[mode][0]).to(device=device, memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        log_fields = (
            "epoch", "train_loss", "validation_macro_f1", "validation_accuracy", "epoch_seconds"
        )
        training_seconds, validation_seconds, optimizer_steps = 0.0, 0.0, 0
        with (run_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=log_fields)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                model.train()
                _synchronize(device)
                started = time.perf_counter()
                loss_total = 0.0
                for indices in torch.randperm(len(train_targets), generator=batches).split(batch_size):
                    logits = model(*_select_views(train_views, mode, indices, device))
                    loss = criterion(logits, train_targets[indices].to(device))
                    if not bool(torch.isfinite(loss)):
                        raise ValueError(f"Non-finite training loss at epoch {epoch}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss.detach()) * len(indices)
                    optimizer_steps += 1
                _synchronize(device)
                epoch_seconds = time.perf_counter() - started
                training_seconds += epoch_seconds
                scores = {"macro_f1": "", "accuracy": ""}
                if clean_validation is not None:
                    output, elapsed = _evaluate(model, mode, clean_validation, classes, batch_size)
                    validation_seconds += elapsed
                    scores = _scores(output["y_true"], output["y_pred"], classes)
                writer.writerow({
                    "epoch": epoch, "train_loss": loss_total / len(train_targets),
                    "validation_macro_f1": scores["macro_f1"],
                    "validation_accuracy": scores["accuracy"], "epoch_seconds": epoch_seconds,
                })
                handle.flush()
                print(f"{config['run_id']} epoch {epoch}/{epochs}: "
                      f"loss={loss_total / len(train_targets):.5f} "
                      f"clean_macro_f1={scores['macro_f1']}", flush=True)

        export_started = time.perf_counter()
        model.eval()
        torch.save({
            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "config": config,
        }, run_dir / "model.pt")
        (run_dir / "features").mkdir()
        metric_rows, prediction_rows, inference_seconds = [], [], {}
        for name, spec in evaluations.items():
            output, elapsed = _evaluate(model, mode, spec, classes, batch_size)
            inference_seconds[name] = elapsed
            np.savez_compressed(run_dir / "features" / f"{name}.npz", **output)
            if spec["split"] == "validation":
                metric_rows.append(_metric_row(config, spec, output, parameters))
                prediction_rows.extend(_prediction_rows(config, spec, output))
        _write_csv(run_dir / "metrics.csv", METRIC_FIELDS, metric_rows)
        _write_csv(run_dir / "predictions.csv", PREDICTION_FIELDS, prediction_rows)
        if model.fusion == "learned":
            intervention_metrics, intervention_predictions = [], []
            for value in (0.0, 0.5, 1.0):
                for spec in evaluations.values():
                    if spec["split"] != "validation":
                        continue
                    output, _ = _evaluate(model, mode, spec, classes, batch_size, gate_override=value)
                    row = _metric_row(config, spec, output, parameters)
                    row["gate_override"] = value
                    intervention_metrics.append(row)
                    intervention_predictions.extend(_prediction_rows(config, spec, output, value))
            _write_csv(run_dir / "intervention_metrics.csv", (*METRIC_FIELDS, "gate_override"),
                       intervention_metrics)
            _write_csv(run_dir / "intervention_predictions.csv", (*PREDICTION_FIELDS, "gate_override"),
                       intervention_predictions)
        _write_json(run_dir / "cost.json", {
            "parameters": parameters, "training_seconds": training_seconds,
            "export_seconds": time.perf_counter() - export_started,
            "inference_seconds": inference_seconds, "training_entries": len(train_targets),
            "optimizer_steps": optimizer_steps, "threads": threads,
            "device": str(device),
            "epoch_validation_seconds": validation_seconds,
            "inference_timing_method": "paired-view selection, device transfer, forward and CPU output collection; batch size from config; CUDA synchronized",
            "training_timing_method": "optimizer epochs excluding epoch validation and export",
        })
        _write_json(run_dir / "status.json", {"status": "complete", "run_id": config["run_id"]})
        artifacts = {
            path.relative_to(run_dir).as_posix(): sha256_file(path)
            for path in sorted(run_dir.rglob("*")) if path.is_file()
        }
        _write_json(run_dir / "completion.json", {
            "schema_version": 1, "status": "complete", "config_fingerprint": config_fingerprint(config),
            "artifacts": artifacts, "evaluations": list(evaluations), "fusion": model.fusion,
        })
        return run_dir
    except BaseException as error:
        _write_json(run_dir / "status.json", {
            "status": "failed", "run_id": config["run_id"],
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


def verify_completed_run(run_dir, config):
    """Refuse incomplete, altered, or differently configured completed runs."""
    run_dir = Path(run_dir)
    try:
        completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        expected = config_fingerprint(config)
        if saved_config != config or completion["config_fingerprint"] != expected:
            raise ValueError("Completed run configuration does not match the requested configuration")
        if completion.get("status") != "complete":
            raise ValueError("Run is not marked complete")
        artifacts = completion["artifacts"]
        required = {
            "config.json", "status.json", "model.pt", "training_log.csv", "metrics.csv",
            "predictions.csv", "cost.json",
        }
        required.update(f"features/{name}.npz" for name in completion["evaluations"])
        if INPUT_MODES[config["input"]][0] == "learned":
            required.update(("intervention_metrics.csv", "intervention_predictions.csv"))
        if not required.issubset(artifacts):
            raise ValueError("Completion manifest is missing required artifacts")
        actual = {
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*") if path.is_file() and path.name != "completion.json"
        }
        if actual != set(artifacts):
            raise ValueError("Run file inventory differs from its completion manifest")
        root = run_dir.resolve()
        for relative, fingerprint in artifacts.items():
            path = (run_dir / relative).resolve()
            if root not in path.parents or not path.is_file() or sha256_file(path) != fingerprint:
                raise ValueError(f"Artifact missing or changed: {relative}")
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ValueError("Run status is not complete")
        return True
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid or incomplete run: {run_dir}: {error}") from error
