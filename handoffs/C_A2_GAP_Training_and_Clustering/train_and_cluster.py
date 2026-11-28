"""
Train and evaluate traffic sign classifiers in a single file.

This script covers A2 mixed augmentation, bilateral filtering, CLAHE, native
ResNet18 GAP, scalar-gated fusion, thirteen-condition evaluation, and feature analysis.
"""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, replace
import hashlib
from itertools import product
import json
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


# ---------------------------------------------------------------------------
# 01 | EXPERIMENT SWITCHES AND SHARED SETTINGS
#
# Use "all" for the full grid or "custom" for the settings in CUSTOM_SWITCHES.
# Command-line flags can set or filter individual switches. With --select,
# the resulting combinations appear as a numbered menu before any data loads.
#
# Bilateral filtering, CLAHE, gated fusion, and A2 affect the training run.
# The cluster switch controls post-training KMeans analysis of that run.
# ---------------------------------------------------------------------------

PACKAGE = Path(__file__).resolve().parent
CLASSES = (1, 2, 3, 4, 5, 7, 8)
SEEDS = (42, 43, 44)
BATCH_SIZE = 128
EPOCHS = 10


@dataclass(frozen=True)
class Switches:
    bilateral: bool = True
    clahe: bool = True
    gate: bool = True
    a2: bool = True
    cluster: bool = True


RUN_GROUP = "all"
CUSTOM_SWITCHES = Switches(bilateral=True, clahe=True, gate=True, a2=True, cluster=True)

# These names map historical checkpoint metadata to the current switches.
PRESETS = {
    "A": Switches(bilateral=False, clahe=False, gate=False, cluster=False),
    "B": Switches(gate=False),
    "C": Switches(),
}
INPUT_MODES = {"A": "raw", "B": "denoised_clahe", "C": "dual_denoised"}
TRAIN_SWITCHES = ("bilateral", "clahe", "gate", "a2")
LEVELS = {
    "motion": (3, 5, 7),
    "noise": (10, 20, 30),
    "dark": (0.70, 0.45, 0.25),
    "jpeg": (40, 20, 10),
}
OFFICIAL_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


# ---------------------------------------------------------------------------
# 02 | FROZEN SEVEN-CLASS SPLIT AND IMAGE LOADING
#
# A capture track contains consecutive frames of the same sign. Select whole
# 30-frame tracks and assign 14 per class to training and 4 to validation.
# This keeps neighboring frames together and yields 2,940 training images and
# 840 validation images. Compare the replayed split against pilot_split.csv
# before reading the images and resizing them to 32 by 32.
# ---------------------------------------------------------------------------

def frozen_rows(data_root, split_csv, rng):
    expected = []
    base = data_root / "GTSRB" / "Final_Training" / "Images"
    for category in CLASSES:
        category_name = f"{category:05d}"
        tracks = defaultdict(list)
        for image in sorted((base / category_name).glob("*.ppm")):
            tracks[image.stem.split("_")[0]].append(image)
        complete = sorted(track for track, images in tracks.items() if len(images) == 30)
        if len(complete) < 18:
            raise ValueError(f"Class {category} requires at least 18 complete tracks")
        for position, track in enumerate(rng.permutation(complete)[:18]):
            split = "train" if position < 14 else "validation"
            expected.extend((split, path.relative_to(data_root).as_posix(), category,
                             f"{category_name}/{track}") for path in tracks[track])
    with split_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["class_id"] = int(row["class_id"])
    actual = [(row["split"], row["image"], row["class_id"], row["track_id"]) for row in rows]
    if actual != expected:
        raise ValueError("The split does not match the frozen seven-class experiment")
    return ([row for row in rows if row["split"] == "train"],
            [row for row in rows if row["split"] == "validation"])


def read_bgr(data_root, row):
    source = np.fromfile(data_root / row["image"], dtype=np.uint8)
    image = cv2.imdecode(source, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {row['image']}")
    return cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)


def corrupt(image, kind, value, *, axis=0, rng=None):
    # Shared pixel transforms for training augmentation and validation.
    # Gaussian noise is clipped back to the 8-bit pixel range. JPEG quality
    # controls an encode/decode round trip so the input contains real artifacts.
    if kind == "motion":
        # A normalized line kernel gives horizontal or vertical motion blur.
        kernel = np.zeros((value, value), dtype=np.float32)
        if axis == 0:
            kernel[value // 2, :] = 1 / value
        else:
            kernel[:, value // 2] = 1 / value
        return cv2.filter2D(image, -1, kernel)
    if kind == "noise":
        return np.clip(image.astype(np.float32) + rng.normal(0, value, image.shape),
                       0, 255).astype(np.uint8)
    if kind == "dark":
        return np.clip(image.astype(np.float32) * value, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, value])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    raise ValueError(f"Unknown corruption: {kind}")


# ---------------------------------------------------------------------------
# 03 | A2 MIXED AUGMENTATION AND THE A0 CONTROL
#
# Choose the same 1,470 source images once for both settings. With A2 on,
# append 490 motion-blurred, 490 Gaussian-noisy, and 490 darkened copies.
# With A2 off, append the corresponding clean copies. Both settings train
# on 4,410 images in the same label order and have the same sample budget.
#
# The data generator starts at seed 42 for every experiment. Model seeds
# control training initialization and shuffling independently of this data.
# ---------------------------------------------------------------------------

def prepare_data(data_root, split_csv):
    rng = np.random.default_rng(42)
    train_rows, valid_rows = frozen_rows(data_root, split_csv, rng)
    train = [read_bgr(data_root, row) for row in train_rows]
    valid = [read_bgr(data_root, row) for row in valid_rows]
    extra_indices = rng.choice(len(train), len(train) // 2, replace=False)

    # Draw a motion kernel and direction for every selected image before
    # assigning corruption types. The draw order reproduces the original A2
    # samples and the validation medium-noise pixels generated afterward.
    motion = [(int(rng.choice(LEVELS["motion"])), int(rng.integers(0, 2)))
              for _ in extra_indices]
    mix_types = rng.permutation(np.resize(np.arange(3), len(extra_indices)))
    extra = []
    for index, kind, (kernel, axis) in zip(extra_indices, mix_types, motion):
        if kind == 0:
            transformed = corrupt(train[index], "motion", kernel, axis=axis)
        elif kind == 1:
            transformed = corrupt(train[index], "noise", float(rng.uniform(8, 24)), rng=rng)
        else:
            transformed = corrupt(train[index], "dark", float(rng.uniform(0.38, 0.70)))
        extra.append(transformed)

    targets = [CLASSES.index(row["class_id"]) for row in train_rows]
    augmented_targets = targets + [targets[index] for index in extra_indices]

    # Continue the A2 random stream for medium validation noise, including
    # when training will use A0. This gives both settings identical input
    # pixels for the medium-noise development condition.
    medium_noise = [corrupt(image, "noise", 20, rng=rng) for image in valid]
    training = {True: train + extra, False: train + [train[index] for index in extra_indices]}
    return train_rows, valid_rows, train, valid, training, augmented_targets, medium_noise


# ---------------------------------------------------------------------------
# 04 | THIRTEEN DEVELOPMENT VALIDATION CONDITIONS
#
# Evaluate each of the same 840 images clean and under three severity levels
# each of motion blur, Gaussian noise, darkening, and JPEG compression.
# Generate all condition pixels once, before model-specific preprocessing,
# so every configuration and seed receives the same underlying disturbance.
# ---------------------------------------------------------------------------

def validation_conditions(clean, rows, medium_noise):
    yield "clean", "none", None, clean
    for kind, values in LEVELS.items():
        for severity, value in zip(("low", "medium", "high"), values):
            if (kind, severity) == ("noise", "medium"):
                yield kind, severity, value, medium_noise
                continue
            images = []
            for index, (image, row) in enumerate(zip(clean, rows)):
                rng = None
                if kind == "noise":
                    # Low/high share the same per-image normal draws at different scales.
                    key = f"20261117:{row['image']}".encode("utf-8")
                    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
                    rng = np.random.default_rng(seed)
                images.append(corrupt(image, kind, value, axis=index % 2, rng=rng))
            yield kind, severity, value, images


# ---------------------------------------------------------------------------
# 05 | IMAGE PREPROCESSING WITH BILATERAL FILTERING AND LAB-L CLAHE
#
# The processed branch applies bilateral filtering first and CLAHE second
# when both are enabled. A single-view model receives the processed image.
# A gated model also receives the original RGB view of the same sample.
# Here "original" means before enhancement, with any existing disturbance
# preserved. With both filters off, the two views are identical.
# ---------------------------------------------------------------------------

def make_views(bgr_images, switches):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) if switches.clahe else None
    raw, processed = [], []
    for image in bgr_images:
        if switches.gate:
            raw.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # ---- BILATERAL FILTER | SMOOTH NOISE WHILE RETAINING EDGES ----
        # Neighbor weights depend on pixel distance and color similarity.
        # d=5 sets the neighborhood diameter. sigmaColor=25 and sigmaSpace=2
        # set the falloff for color differences and spatial distances.
        if switches.bilateral:
            image = cv2.bilateralFilter(image, d=5, sigmaColor=25, sigmaSpace=2)

        # ---- CLAHE | INCREASE LOCAL CONTRAST ON THE LAB L CHANNEL ----
        # Use a 4 by 4 tile grid and clipLimit=2.0 to limit contrast amplification.
        # Apply CLAHE to L only, keeping the A and B channels from the LAB
        # conversion. Convert the resulting image to RGB for ResNet18.
        if clahe is not None:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed.append(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        else:
            processed.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Produce float RGB batches shaped [N, 3, H, W] with values in [0, 1].
    # The model applies the official ImageNet transforms after this step.
    return tuple(
        torch.from_numpy(np.stack(stack)).permute(0, 3, 1, 2).float().div_(255)
        .contiguous(memory_format=torch.channels_last)
        for stack in ((raw, processed) if switches.gate else (processed,))
    )


# ---------------------------------------------------------------------------
# 06 | OFFICIAL RESNET18, NATIVE GAP, AND LEARNED DUAL-VIEW GATING
#
# Initialize ResNet18 with the official ImageNet weights and keep its native
# global average pooling (GAP). Replace the 1,000-class head with a shared
# seven-class classifier applied to the final 512-dimensional representation.
#
# In dual-view mode, both images use the same encoder. The gate reads both
# 512-value feature vectors and predicts one weight g for the processed view.
# g = sigmoid(W * concat(raw, processed) + b)
# Fuse the vectors as (1 - g) * raw + g * processed, then classify that vector.
# ---------------------------------------------------------------------------

class ResNet18Fusion(nn.Module):
    def __init__(self, group, imagenet_state=None, *, switches=None):
        super().__init__()
        self.group = group
        self.switches = PRESETS[group] if switches is None else switches
        # Load the original 1,000-class weights before replacing the original head.
        self.encoder = resnet18(weights=None)
        if imagenet_state is not None:
            self.encoder.load_state_dict(imagenet_state, strict=True)
        self.encoder.fc = nn.Identity()  # Leave native GAP intact; each vector has 512 values.
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.classifier = nn.Linear(512, len(CLASSES))

        # ---- GATE INITIALIZATION | START WITH EQUAL WEIGHT ON BOTH VIEWS ----
        # The linear layer reads 1,024 concatenated features. Zero weights and
        # bias give sigmoid(0) = 0.5 at initialization. Backpropagation through
        # the classification loss learns the weights and bias during training.
        self.gate = nn.Linear(1024, 1) if self.switches.gate else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward_details(self, view, enhanced=None, gate_override=None):
        if self.gate is None:
            # A single branch uses whichever preprocessing switches are enabled.
            features = self.encoder(self.preprocess(view))
            return {"logits": self.classifier(features), "features": features,
                    "raw_features": features, "gate": features.new_full((len(view), 1), float("nan"))}

        if enhanced is None or enhanced.shape != view.shape:
            raise ValueError("Gated fusion requires paired views with the same shape")
        # ---- SHARED ENCODING AND PER-IMAGE FEATURE FUSION ----
        # Stack the two views along the batch axis for one encoder call, then
        # split the embeddings back into raw and processed pairs. Concatenate
        # each pair along the feature axis for the linear gate and sigmoid.
        raw, filtered = self.encoder(self.preprocess(torch.cat((view, enhanced)))).chunk(2)
        if gate_override is None:
            g = self.gate(torch.cat((raw, filtered), dim=1)).sigmoid()
        else:
            # For an inference intervention, set g=0 for raw features, 0.5 for
            # equal weighting, or 1 for processed features with the same model.
            g = raw.new_full((len(raw), 1), gate_override)
        features = torch.lerp(raw, filtered, g)  # (1 - g) * raw + g * enhanced
        return {"logits": self.classifier(features), "features": features,
                "raw_features": raw, "enhanced_features": filtered, "gate": g}

    def forward(self, view, enhanced=None):
        return self.forward_details(view, enhanced)["logits"]


def select_views(views, indices, device):
    # Use the same minibatch indices for both views so image pairs stay aligned.
    return tuple(view[indices].to(device, memory_format=torch.channels_last) for view in views)


# ---------------------------------------------------------------------------
# 07 | TRAINING, CLASSIFICATION LOSS, AND CHECKPOINT RELOAD
#
# Each configuration and seed starts from the official ImageNet initialization.
# Use shuffled batches of 128, ten epochs, and AdamW at a learning rate of
# 0.003. Cross-entropy supervises the seven classes and updates the encoder,
# classifier, and gate when present. Save the model from the final epoch.
#
# Checkpoint evaluation reconstructs the input switches and loads the saved
# weights. Historical A/B/C checkpoint names map to the same model layouts.
# ---------------------------------------------------------------------------

def train_model(group, switches, train_images, targets, imagenet_weights, seed, device):
    if hashlib.sha256(imagenet_weights.read_bytes()).hexdigest() != OFFICIAL_SHA256:
        raise ValueError("The ImageNet initialization must be the project's official weight file")
    # Seed initialization and the independent minibatch sampler for this run.
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    batches = torch.Generator().manual_seed(seed)
    state = torch.load(imagenet_weights, map_location="cpu", weights_only=True)
    model = ResNet18Fusion(group, state, switches=switches).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    views = make_views(train_images, switches)
    target = torch.as_tensor(targets, dtype=torch.long)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for indices in torch.randperm(len(target), generator=batches).split(BATCH_SIZE):
            # The same classification loss updates the encoder, classifier, and any gate.
            logits = model(*select_views(views, indices, device))
            loss = criterion(logits, target[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(indices)
        print(f"Epoch {epoch}/{EPOCHS}: loss={total_loss / len(target):.5f}", flush=True)
    return model  # Report the final epoch, without choosing an epoch on the validation scores.


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    augmentation = config["augmentation"]
    if config.get("pooling", "gap") != "gap" or augmentation not in ("mixed", "repeat_clean"):
        raise ValueError("Use a GAP checkpoint with a supported augmentation strategy")
    if "switches" in config:
        group = config["group"]
        switches = Switches(**config["switches"])
    else:
        # Old A/B/C checkpoints did not record switches explicitly.
        group = next((group for group, mode in INPUT_MODES.items() if mode == config["input"]), None)
        if group is None:
            raise ValueError("The checkpoint must use a known A/B/C input")
        switches = replace(PRESETS[group], a2=augmentation == "mixed")
    if config["seed"] not in SEEDS or switches.a2 != (augmentation == "mixed"):
        raise ValueError("The checkpoint's training seed or augmentation is inconsistent")
    model = ResNet18Fusion(group, switches=switches)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device, memory_format=torch.channels_last), config["seed"]


# ---------------------------------------------------------------------------
# 08 | CLASSIFICATION INFERENCE AND PER-IMAGE FEATURES
#
# Rebuild the input views using the model's switches and run inference with
# the final weights. Sort exported examples by image ID so the same sample
# can be matched across conditions, models, and seeds.
#
# The classifier predicts one of seven classes from the final embedding.
# Export that embedding, branch features, gate values when available, and
# the true and predicted class IDs for later analysis.
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, images, rows, device, gate_override=None):
    # Interference is generated on the frozen row order; outputs are sorted by image ID.
    views = make_views(images, model.switches)
    order = sorted(range(len(rows)), key=lambda index: rows[index]["image"])
    model.eval()
    collected = defaultdict(list)
    for indices in torch.tensor(order, dtype=torch.long).split(BATCH_SIZE):
        details = model.forward_details(*select_views(views, indices, device),
                                        gate_override=gate_override)
        predicted = np.asarray(CLASSES)[details.pop("logits").argmax(1).cpu().numpy()]
        collected["y_pred"].append(predicted)
        for name, value in details.items():
            collected[name].append(value.cpu().numpy())
    output = {name: np.concatenate(values) for name, values in collected.items()}
    output.update(image=np.asarray([rows[i]["image"] for i in order]),
                  track_id=np.asarray([rows[i]["track_id"] for i in order]),
                  y_true=np.asarray([rows[i]["class_id"] for i in order]))
    return output


def classification_scores(result):
    # Accuracy counts correct predictions. Macro-F1 averages the seven
    # class-wise F1 scores equally, using the same classifier predictions.
    y_true, y_pred = result["y_true"], result["y_pred"]
    return {"accuracy": np.mean(y_true == y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)}


# ---------------------------------------------------------------------------
# 09 | KMEANS CLUSTERING AND PCA ON MODEL FEATURES
#
# L2-normalize the 512-dimensional clean training embeddings. Fit a 2D PCA
# basis for visualization and, with clustering enabled, seven KMeans
# centroids in the normalized 512-dimensional space. Each fitted analysis
# is reused across all thirteen validation conditions for this run.
#
# ARI and NMI compare cluster assignments with the true labels. Same-cluster
# rate checks how often each disturbed sample keeps its clean cluster ID.
# All of this runs after training. KMeans on and off can share the same
# weights and classification scores.
# ---------------------------------------------------------------------------

def analyze_features(switches, train_clean, validation):
    with threadpool_limits(limits=2):
        train = normalize(train_clean["features"])
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42).fit(train)

        # Fit the centroids without labels, using 20 initializations and a
        # fixed seed. When clustering is off, keep the PCA projections.
        kmeans = None
        if switches.cluster:
            kmeans = KMeans(n_clusters=7, n_init=20, random_state=42).fit(train)
        clean = validation[("clean", "none")]
        clean_clusters = None if kmeans is None else kmeans.predict(normalize(clean["features"]))
        metrics, projections = [], {}
        for (condition, severity), result in validation.items():
            if not np.array_equal(clean["image"], result["image"]):
                raise ValueError("Validation conditions must use the same ordered image IDs")
            features = normalize(result["features"])
            projections[f"{condition}_{severity}"] = pca.transform(features)
            if kmeans is None:
                continue
            clusters = kmeans.predict(features)
            # Use class labels for scoring after assigning each example to a cluster.
            metrics.append({
                "condition": condition, "severity": severity,
                "ari": adjusted_rand_score(result["y_true"], clusters),
                "nmi": normalized_mutual_info_score(result["y_true"], clusters,
                                                    average_method="arithmetic"),
                "same_cluster": np.mean(clusters == clean_clusters),
            })
    return metrics, projections


# ---------------------------------------------------------------------------
# 10 | METRIC AGGREGATION, FEATURE EXPORTS, AND PCA PLOTS
#
# Average conditions equally within each run, then report the mean and
# sample standard deviation across seeds. Keep the all-13 mean and the
# 12-corruption mean separately so the effect of including clean is visible.
#
# Keep per-condition NPZ files so each result can be recomputed from aligned
# examples. The PCA overview shows clean and the four high-severity cases.
# Each row uses its own PCA basis fitted on that model's clean train features.
# ---------------------------------------------------------------------------

def condition_means(rows):
    # The twelve-condition mean excludes clean; the thirteen-condition mean includes it.
    metrics = set(rows[0]) - {"condition", "severity", "group", "seed"}
    return {bucket: {name: float(np.mean([row[name] for row in selected])) for name in sorted(metrics)}
            for bucket, selected in (("all_13", rows),
                                     ("corrupt_all", [r for r in rows if r["condition"] != "clean"]))}


def summarize_seeds(rows_by_seed):
    # Conditions have equal weight within a run; run means have equal weight across seeds.
    means = [condition_means(rows) for rows in rows_by_seed.values()]
    summary = {}
    for bucket, metrics in means[0].items():
        summary[bucket] = {}
        for name in metrics:
            values = [run[bucket][name] for run in means]
            summary[bucket][name] = {"mean": float(np.mean(values)),
                                     "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return summary


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_run(model, seed, device, output, train, train_rows, conditions, valid_rows):
    # Export the training split's original images with the model's selected
    # views. These embeddings are the reference set for both PCA and KMeans.
    train_clean = evaluate(model, train, train_rows, device)
    np.savez_compressed(output / "train_clean.npz", **train_clean)
    validation, scores = {}, []
    for condition, severity, _, images in conditions:
        result = evaluate(model, images, valid_rows, device)
        validation[(condition, severity)] = result
        np.savez_compressed(output / f"{condition}_{severity}.npz", **result)
        scores.append({"group": model.group, "seed": seed, "condition": condition,
                       "severity": severity, **classification_scores(result)})
        print(f"{model.group} seed={seed} {condition}/{severity}: "
              f"accuracy={scores[-1]['accuracy']:.2%}", flush=True)
    clusters, projections = analyze_features(model.switches, train_clean, validation)
    clean = validation[("clean", "none")]
    np.savez_compressed(output / "pca_projections.npz",
                        image=clean["image"], y_true=clean["y_true"], **projections)
    return scores, [{"group": model.group, "seed": seed, **row} for row in clusters], (
        clean["y_true"], projections)


def save_pca_comparison(output, pca_by_group, seed):
    # Colors show true classes. Each row shares one PCA basis fitted to that model's training set.
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    panels = ("clean_none", "motion_high", "noise_high", "dark_high", "jpeg_high")
    titles = ("Clean", "High motion blur", "High Gaussian noise", "High darkening", "High JPEG")
    figure, axes = plt.subplots(len(pca_by_group), len(panels), figsize=(15, 2.7 * len(pca_by_group) + 1),
                                sharex="row", sharey="row", squeeze=False)
    colors = plt.colormaps["tab10"].colors
    for row, (group, (labels, projections)) in enumerate(pca_by_group.items()):
        for column, (condition, title) in enumerate(zip(panels, titles)):
            ax = axes[row, column]
            coordinates = projections[condition]
            for category, speed, color in zip(CLASSES, (20, 30, 50, 60, 70, 80, 100), colors):
                points = coordinates[labels == category]
                ax.scatter(points[:, 0], points[:, 1], color=color, s=9, linewidths=0,
                           alpha=0.65, label=f"{speed} km/h")
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(f"{group} / PC2")
            if row == len(pca_by_group) - 1:
                ax.set_xlabel("PC1")
            ax.locator_params(axis="x", nbins=4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    figure.suptitle(f"GAP training-feature PCA, model seed {seed}")
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(output / f"pca_seed{seed}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 11 | FULL SWITCH GRID AND INTERACTIVE EXPERIMENT SELECTION
#
# Four training switches give 16 model configurations. Crossing them with
# KMeans on/off gives 32 result rows. Train each model configuration once
# per seed, then share its scores between the requested KMeans settings.
#
# The optional --select menu numbers the rows after applying CLI filters.
# Selected rows are grouped by training configuration before training starts.
# ---------------------------------------------------------------------------

def model_id(switches):
    # Keep the model ID independent of the reporting-only KMeans switch.
    return f"b{int(switches.bilateral)}_c{int(switches.clahe)}_g{int(switches.gate)}_a{int(switches.a2)}"


def experiment_plan(group, overrides):
    if group == "all":
        axes = [(overrides[name],) if name in overrides else (False, True)
                for name in TRAIN_SWITCHES]
        cluster_options = (overrides["cluster"],) if "cluster" in overrides else (False, True)
        experiments = []
        for flags in product(*axes):
            switches = Switches(*flags, cluster=any(cluster_options))
            experiments.append((model_id(switches), switches, cluster_options))
        return experiments
    switches = replace(CUSTOM_SWITCHES, **overrides)
    return [(model_id(switches), switches, (switches.cluster,))]


def select_experiments(experiments, parser):
    choices = [(group, switches, cluster) for group, switches, clusters in experiments
               for cluster in clusters]
    print("\n  #  Bilateral  CLAHE  Dual-view gate  A2 mixed  KMeans")
    for index, (_, switches, cluster) in enumerate(choices, start=1):
        flags = (switches.bilateral, switches.clahe, switches.gate, switches.a2, cluster)
        values = ["on" if flag else "off" for flag in flags]
        print(f"{index:3}  {values[0]:<9}  {values[1]:<5}  {values[2]:<14}  {values[3]:<8}  {values[4]}")
    while True:
        try:
            selection = input("\nChoose numbers (e.g. 1, 4, 7), Enter for all, or q to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            parser.exit(1, "\nSelection cancelled.\n")
        if selection.lower() == "q":
            parser.exit(0, "Selection cancelled.\n")
        if not selection:
            return experiments
        try:
            indices = {int(value) for value in selection.replace(",", " ").split()}
        except ValueError:
            indices = set()
        if indices and all(1 <= index <= len(choices) for index in indices):
            break
        print(f"Enter one or more numbers from 1 to {len(choices)}.")

    selected = {}
    for index in sorted(indices):
        group, switches, cluster = choices[index - 1]
        selected.setdefault(group, (switches, []))[1].append(cluster)
    return [(group, replace(switches, cluster=any(clusters)), tuple(clusters))
            for group, (switches, clusters) in selected.items()]


# ---------------------------------------------------------------------------
# 12 | RUN THE EXPERIMENT AND WRITE THE DELIVERY FILES
#
# Resolve the requested combinations before loading data. Train each selected
# configuration and seed, or evaluate an existing checkpoint. Save weights,
# per-condition metrics, and per-image features. Record the switches and
# pixel hashes so model and input settings can be verified from the outputs.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GAP experiments with independent switches")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=PACKAGE / "pilot_split.csv")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path, help="Final checkpoint from an existing run")
    weights.add_argument("--imagenet-weights", type=Path, help="Official initialization for training")
    parser.add_argument("--group", choices=("all", "custom"), default=RUN_GROUP,
                        help="Full matrix or one custom setting")
    parser.add_argument("--select", action="store_true", help="Choose experiment combinations at startup")
    for name, help_text in (
        ("bilateral", "Bilateral filtering"),
        ("clahe", "CLAHE on the LAB luminance channel"),
        ("gate", "Raw and processed views with learned gating; off uses one view"),
        ("a2", "Mixed augmentation; off repeats the same original images (A0)"),
        ("cluster", "KMeans analysis; PCA and classification are always exported"),
    ):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None, help=help_text)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Each training seed may be supplied only once")
    overrides = {name: getattr(args, name) for name in (*TRAIN_SWITCHES, "cluster")
                 if getattr(args, name) is not None}
    if args.checkpoint:
        if args.select or any(name in overrides for name in TRAIN_SWITCHES):
            parser.error("Model and augmentation switches come from the checkpoint")
    else:
        experiments = experiment_plan(args.group, overrides)
        if args.select:
            experiments = select_experiments(experiments, parser)
        seeds = args.seeds
        figure_seed = 42 if 42 in seeds else seeds[0]
        combination_count = sum(len(clusters) for _, _, clusters in experiments)
        print(f"Selected {combination_count} result combination(s). "
              f"Training {len(experiments)} configuration(s) across {len(seeds)} seed(s) "
              f"= {len(experiments) * len(seeds)} run(s).", flush=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch runtime")

    # Match the reference run's numerical settings. Select deterministic
    # PyTorch operations, disable cuDNN benchmarking, and turn off TF32.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cv2.setNumThreads(1)
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    train_rows, valid_rows, train, valid, training, targets, medium_noise = prepare_data(
        args.data_root, args.split)
    conditions = tuple(validation_conditions(valid, valid_rows, medium_noise))
    args.output.mkdir(parents=True)
    pca_by_group, classification, clustering, summaries = {}, [], [], {}
    combinations = []
    if args.checkpoint:
        model, seed = load_checkpoint(args.checkpoint, args.device)
        switches = replace(model.switches, **overrides)
        model.switches = switches
        experiments, seeds = [(model.group, switches, (switches.cluster,))], (seed,)
        figure_seed = seed
    weights_path = args.imagenet_weights or PACKAGE / "weights" / "resnet18-f37072fd.pth"

    # Start a separate run per configuration and seed. The A2 switch picks
    # the training input set, while every run shares all validation pixels.
    for group, switches, cluster_options in experiments:
        mode = INPUT_MODES.get(group, group)
        augmentation = "mixed" if switches.a2 else "repeat_clean"
        classification_by_seed, clustering_by_seed = {}, {}
        for seed in seeds:
            run_dir = args.output / "runs" / f"{augmentation}_{mode}_seed{seed}"
            run_dir.mkdir(parents=True)
            if not args.checkpoint:
                model = train_model(group, switches, training[switches.a2], targets,
                                    weights_path, seed, args.device)
                torch.save({"config": {"input": mode, "group": group, "seed": seed,
                                       "augmentation": augmentation, "pooling": "gap",
                                       "switches": vars(switches)},
                            "model_state": {key: value.detach().cpu()
                                            for key, value in model.state_dict().items()}}, run_dir / "model.pt")
            scores, clusters, pca = export_run(
                model, seed, args.device, run_dir, train, train_rows, conditions, valid_rows)
            classification_by_seed[seed] = scores
            classification.extend(scores)
            if clusters:
                clustering_by_seed[seed] = clusters
                clustering.extend(clusters)
            if seed == figure_seed:
                pca_by_group[group] = pca
            del model
        scores = summarize_seeds(classification_by_seed)
        cluster_scores = summarize_seeds(clustering_by_seed) if clustering_by_seed else {}

        # Write one result row per requested KMeans setting. Both rows share
        # the same classification summary. Cluster metrics go in the enabled row.
        for cluster_enabled in cluster_options:
            name = f"{group}_k{int(cluster_enabled)}"
            options = replace(switches, cluster=cluster_enabled)
            metrics = cluster_scores["all_13"] if cluster_enabled else {}
            combinations.append({"combination": name, "model_group": group, "seeds": len(seeds),
                                 **vars(options), "accuracy_13": scores["all_13"]["accuracy"]["mean"],
                                 "accuracy_12": scores["corrupt_all"]["accuracy"]["mean"],
                                 "macro_f1_13": scores["all_13"]["macro_f1"]["mean"],
                                 "macro_f1_12": scores["corrupt_all"]["macro_f1"]["mean"],
                                 "ari_13": metrics.get("ari", {}).get("mean", ""),
                                 "nmi_13": metrics.get("nmi", {}).get("mean", ""),
                                 "same_cluster_13": metrics.get("same_cluster", {}).get("mean", "")})
            summaries[name] = {"switches": vars(options), "classification": scores}
            if cluster_enabled:
                summaries[name]["clustering"] = cluster_scores

    # The shared manifest records exact condition pixels for comparison against a reference run.
    manifest = [
        {"condition": kind, "severity": level, "parameter": value, "n_images": len(images),
         "pixel_sha256": hashlib.sha256(np.ascontiguousarray(images, dtype=np.uint8).tobytes()).hexdigest()}
        for kind, level, value, images in conditions
    ]
    write_csv(args.output / "classification.csv", classification)
    if clustering:
        write_csv(args.output / "clustering.csv", clustering)
    write_csv(args.output / "combinations.csv", combinations)
    if len(pca_by_group) > 3:
        # Keep the overview legible. Every run still exports its complete PCA coordinates.
        pca_by_group = dict(list(pca_by_group.items())[:3])
    save_pca_comparison(args.output, pca_by_group, figure_seed)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (args.output / "conditions.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
