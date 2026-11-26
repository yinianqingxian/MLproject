"""
用途：以官方预训练 ResNet18 替换共享骨干并保留既有双视图融合实验。
核心职责：
- 对每批 32 像素视图应用官方预处理，提取 512 维特征并执行原有门控。
- 校验本地官方权重及配置来源，保存兼容旧字段的最终轮训练产物。
关键副作用/安全约束：
- 不隐式下载权重，仅接受项目内经完整 SHA256 校验的官方文件。
- 仅写新的运行目录，不覆盖旧实验，不读取官方测试集。
"""

import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

if __package__:
    from .cnn_robustness_study import (
        CLASSES, INPUT_MODES, METRIC_FIELDS, PREDICTION_FIELDS, _evaluate, _metric_row,
        _prediction_rows, _scores, _select_views, _synchronize, _validate_views,
        _write_csv, _write_json, config_fingerprint, sha256_file, tensor_views,
        verify_completed_run,
    )
else:
    from cnn_robustness_study import (
        CLASSES, INPUT_MODES, METRIC_FIELDS, PREDICTION_FIELDS, _evaluate, _metric_row,
        _prediction_rows, _scores, _select_views, _synchronize, _validate_views,
        _write_csv, _write_json, config_fingerprint, sha256_file, tensor_views,
        verify_completed_run,
    )


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = PROJECT / "assets/weights/resnet18-f37072fd.pth"
WEIGHTS_ID = "ResNet18_Weights.IMAGENET1K_V1"
OFFICIAL_WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
BACKBONE = "torchvision.models.resnet18"


def _torchvision():
    try:
        import torchvision
    except (ImportError, RuntimeError, OSError) as error:
        raise RuntimeError("ResNet18 requires a matched torch/torchvision installation") from error
    return torchvision


def resolve_weights_path(weights_path=None):
    """Resolve a project-local official checkpoint without any network access."""
    candidate = DEFAULT_WEIGHTS_PATH if weights_path is None else Path(weights_path)
    if not candidate.is_absolute():
        candidate = PROJECT / candidate
    candidate = candidate.resolve()
    if PROJECT.resolve() not in candidate.parents:
        raise ValueError("ResNet18 weights must be stored inside the project")
    if not candidate.is_file():
        raise ValueError(f"Local official ResNet18 weights are missing: {candidate}")
    if sha256_file(candidate) != OFFICIAL_WEIGHTS_SHA256:
        raise ValueError("ResNet18 weights do not match the official ImageNet checkpoint SHA256")
    return candidate


def weights_metadata(pretrained=True, weights_path=None):
    """Return verified provenance; untrained engineering runs make no weight claim."""
    if not isinstance(pretrained, bool):
        raise ValueError("pretrained must be an explicit boolean")
    if not pretrained and weights_path is not None:
        raise ValueError("Untrained smoke runs cannot declare a pretrained weights path")
    vision = _torchvision()
    path = resolve_weights_path(weights_path) if pretrained else None
    return {
        "backbone": BACKBONE, "feature_dim": 512, "network_input_size": 224,
        "pretrained": pretrained, "weights_id": WEIGHTS_ID if pretrained else None,
        "weights_sha256": sha256_file(path) if path is not None else None,
        "weights_path": path.relative_to(PROJECT.resolve()).as_posix() if path is not None else None,
        "torchvision_version": str(vision.__version__),
        "torchvision_module_path": str(Path(vision.__file__).resolve()),
    }


def _validate_weight_config(config):
    required = {
        "backbone", "feature_dim", "network_input_size", "pretrained", "weights_id",
        "weights_sha256", "weights_path", "torchvision_version", "torchvision_module_path",
    }
    if not required.issubset(config):
        raise ValueError(f"Missing ResNet18 provenance fields: {sorted(required - set(config))}")
    if config["weights_path"] is not None:
        path = Path(config["weights_path"])
        if path.is_absolute() or "\\" in config["weights_path"] or ".." in path.parts:
            raise ValueError("Recorded weights_path must be a project-relative POSIX path")
    expected = weights_metadata(config["pretrained"], config["weights_path"])
    for name, value in expected.items():
        if config[name] != value:
            raise ValueError(f"ResNet18 provenance mismatch: {name}")


class ResNet18Fusion(nn.Module):
    """Official shared ResNet18 followed by the existing seven-class fusion head."""

    feature_dim = 512

    def __init__(self, fusion="single", pretrained=True, weights_path=None):
        super().__init__()
        if fusion not in ("single", "learned", "fixed", "spatial"):
            raise ValueError(f"Unknown fusion mode: {fusion}")
        if not isinstance(pretrained, bool):
            raise ValueError("pretrained must be an explicit boolean")
        if not pretrained and weights_path is not None:
            raise ValueError("Untrained smoke runs cannot declare a pretrained weights path")
        vision = _torchvision()
        self.fusion = fusion
        self.preprocess = vision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.encoder = vision.models.resnet18(weights=None)
        if pretrained:
            state = torch.load(resolve_weights_path(weights_path), map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state, strict=True)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, len(CLASSES))
        if fusion == "learned":
            # One weight per image, computed from the pooled 1024-dimensional vector.
            self.gate = nn.Linear(self.feature_dim * 2, 1)
        elif fusion == "spatial":
            # One weight per position: a 1x1 convolution over the [N, 1024, 7, 7] map.
            self.gate = nn.Conv2d(self.feature_dim * 2, 1, kernel_size=1)
        else:
            self.gate = None
        if self.gate is not None:
            # Zero initialisation keeps the initial weight map at exactly 0.5 everywhere.
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def _layer4_map(self, batch):
        """Encoder output before global average pooling: [N, feature_dim, 7, 7]."""
        encoder = self.encoder
        x = self.preprocess(batch)
        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x = encoder.relu(x)
        x = encoder.maxpool(x)
        x = encoder.layer1(x)
        x = encoder.layer2(x)
        x = encoder.layer3(x)
        return encoder.layer4(x)

    def _spatial_details(self, image, enhanced, gate_override, spatial_average):
        if enhanced is None or image.shape != enhanced.shape:
            raise ValueError("Fusion requires paired views of equal shape")
        raw_map, enhanced_map = self._layer4_map(torch.cat((image, enhanced))).chunk(2)
        weight_map = torch.sigmoid(self.gate(torch.cat((raw_map, enhanced_map), dim=1)))
        if gate_override is not None:
            value = float(gate_override)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Gate override must be finite and within [0, 1]")
            weight_map = torch.full_like(weight_map, value)
        elif spatial_average:
            # Intervention: replace every position by that image's own mean weight.
            weight_map = weight_map.mean(dim=(2, 3), keepdim=True).expand_as(weight_map)
        features = ((1 - weight_map) * raw_map + weight_map * enhanced_map).mean(dim=(2, 3))
        return {
            "logits": self.classifier(features),
            "features": features,
            "raw_features": raw_map.mean(dim=(2, 3)),
            "enhanced_features": enhanced_map.mean(dim=(2, 3)),
            "gate": weight_map.mean(dim=(2, 3)).reshape(len(weight_map), 1),
            "gate_map": weight_map,
        }

    def forward_details(self, image, enhanced=None, gate_override=None, spatial_average=False):
        if self.fusion == "spatial":
            return self._spatial_details(image, enhanced, gate_override, spatial_average)
        if self.fusion == "single":
            if enhanced is not None or gate_override is not None:
                raise ValueError("A single branch accepts one view and no gate override")
            features = self.encoder(self.preprocess(image))
            return {
                "logits": self.classifier(features), "features": features,
                "raw_features": features,
                "gate": features.new_full((len(features), 1), float("nan")),
            }
        if enhanced is None or image.shape != enhanced.shape:
            raise ValueError("Fusion requires paired views of equal shape")
        raw_features, enhanced_features = self.encoder(
            self.preprocess(torch.cat((image, enhanced)))
        ).chunk(2)
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
            "logits": self.classifier(features), "features": features,
            "raw_features": raw_features, "enhanced_features": enhanced_features, "gate": weight,
        }

    def forward(self, image, enhanced=None, gate_override=None):
        return self.forward_details(image, enhanced, gate_override)["logits"]


def fit_run(run_dir, config, train_views, train_targets, evaluations):
    """Train the final epoch and export the same schemas with 512-dimensional features."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    config = json.loads(json.dumps(config, allow_nan=False))
    _validate_weight_config(config)
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
        model = ResNet18Fusion(
            INPUT_MODES[mode][0], pretrained=config["pretrained"], weights_path=config["weights_path"]
        ).to(device=device, memory_format=torch.channels_last)
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
            "device": str(device), "epoch_validation_seconds": validation_seconds,
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
