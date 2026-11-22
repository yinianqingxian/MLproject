"""
用途：复跑 GTSRB 七类限速牌的小 CNN 鲁棒性对照实验。
核心职责：
- 重建按拍摄轨迹分组的试验划分和固定的干扰样本。
- 比较相同增强预算下原图、CLAHE 图和双分支融合的表现。
关键副作用/安全约束：
- 在指定输出目录写入划分清单及评估指标。
"""

import argparse
from collections import defaultdict
import csv
from io import StringIO
from pathlib import Path

import numpy as np


CLASSES = (1, 2, 3, 4, 5, 7, 8)
SPLIT_SEED = 42
IMAGE_SIZE = 32
BATCH_SIZE = 128


def select_split(data_root, rng):
    train_folder = data_root / "GTSRB" / "Final_Training" / "Images"
    rows = []
    for class_id in CLASSES:
        tracks = defaultdict(list)
        for image in sorted((train_folder / f"{class_id:05d}").glob("*.ppm")):
            track = image.stem.split("_")[0]
            tracks[track].append(image)
        complete_tracks = sorted(track for track, images in tracks.items() if len(images) == 30)
        if len(complete_tracks) < 18:
            raise ValueError(f"Class {class_id} needs at least 18 complete tracks")
        for position, track in enumerate(rng.permutation(complete_tracks)[:18]):
            split = "train" if position < 14 else "validation"
            track_id = f"{class_id:05d}/{track}"
            for image in tracks[track]:
                rows.append((split, image.relative_to(data_root).as_posix(), class_id, track_id))
    train_tracks = {row[3] for row in rows if row[0] == "train"}
    validation_tracks = {row[3] for row in rows if row[0] == "validation"}
    if train_tracks & validation_tracks or (len(train_tracks), len(validation_tracks)) != (98, 28):
        raise ValueError("Unexpected overlap or number of training and validation tracks")
    return rows


def motion_blur(cv2, image, size, axis):
    kernel = np.zeros((size, size), dtype=np.float32)
    if axis == 0:
        kernel[size // 2, :] = 1 / size
    else:
        kernel[:, size // 2] = 1 / size
    return cv2.filter2D(image, -1, kernel)


def gaussian_noise(image, sigma, rng):
    return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


def darken(image, gain):
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def jpeg_compress(cv2, image):
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 20])
    if not ok:
        raise ValueError("Could not encode a JPEG validation example")
    return cv2.imdecode(payload, cv2.IMREAD_COLOR)


def prepare_images(cv2, data_root, rows, rng):
    images = {"train": [], "validation": []}
    labels = {"train": [], "validation": []}
    for split, path, class_id, _ in rows:
        source = np.fromfile(data_root / path, dtype=np.uint8)
        image = cv2.imdecode(source, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        images[split].append(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA))
        labels[split].append(CLASSES.index(class_id))

    train = images["train"]
    valid = images["validation"]
    extra_indices = rng.choice(len(train), len(train) // 2, replace=False)
    extra_labels = [labels["train"][i] for i in extra_indices]
    targeted = [
        motion_blur(cv2, train[i], int(rng.choice((3, 5, 7))), int(rng.integers(0, 2)))
        for i in extra_indices
    ]
    mixed = []
    mix_types = rng.permutation(np.resize(np.arange(3), len(extra_indices)))
    for position, (index, kind) in enumerate(zip(extra_indices, mix_types)):
        if kind == 0:
            mixed.append(targeted[position])
        elif kind == 1:
            mixed.append(gaussian_noise(train[index], float(rng.uniform(8, 24)), rng))
        else:
            mixed.append(darken(train[index], float(rng.uniform(0.38, 0.7))))

    validation = {
        "clean": valid,
        "motion": [motion_blur(cv2, image, 5, i % 2) for i, image in enumerate(valid)],
        "noise": [gaussian_noise(image, 20, rng) for image in valid],
        "dark": [darken(image, 0.45) for image in valid],
        "jpeg": [jpeg_compress(cv2, image) for image in valid],
    }
    training = {"targeted_motion": train + targeted, "mixed": train + mixed}
    return training, labels["train"] + extra_labels, validation, labels["validation"]


def run_experiments(args, rows, rng):
    import cv2
    from sklearn.metrics import f1_score
    import torch
    from torch import nn

    class SmallCNN(nn.Module):
        def __init__(self, fusion):
            super().__init__()
            layers = []
            for channels_in, channels_out in ((3, 12), (12, 24), (24, 32)):
                layers.extend((
                    nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
                    nn.GroupNorm(4, channels_out),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ))
            self.encoder = nn.Sequential(*layers, nn.Flatten(), nn.Linear(32 * 4 * 4, 32), nn.ReLU())
            self.classifier = nn.Linear(32, len(CLASSES))
            self.gate = nn.Linear(64, 1) if fusion else None
            if self.gate is not None:
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)

        def forward(self, image, enhanced=None):
            if enhanced is None:
                return self.classifier(self.encoder(image))
            raw_feature, enhanced_feature = self.encoder(torch.cat((image, enhanced))).chunk(2)
            weight = self.gate(torch.cat((raw_feature, enhanced_feature), dim=1)).sigmoid()
            return self.classifier(torch.lerp(raw_feature, enhanced_feature, weight))

    def tensor_views(images):
        raw_rgb = []
        enhanced_rgb = []
        for image in images:
            raw_rgb.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_rgb.append(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        return tuple(torch.from_numpy(np.stack(view)).permute(0, 3, 1, 2).float().div_(255)
                     for view in (raw_rgb, enhanced_rgb))

    def select_views(views, mode, indices):
        if mode == "raw":
            return (views[0][indices],)
        if mode == "clahe":
            return (views[1][indices],)
        return tuple(view[indices] for view in views)

    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    training, train_y, validation, valid_y = prepare_images(cv2, args.data_root, rows, rng)
    if (len(train_y), len(valid_y)) != (4410, 840):
        raise ValueError("Unexpected augmentation budget or validation sample count")
    train_targets = torch.tensor(train_y)
    validation_views = {name: tensor_views(images) for name, images in validation.items()}
    criterion = nn.CrossEntropyLoss()
    metrics_path = args.output_dir / "cnn_pilot_metrics.csv"
    print(f"126 tracks, 2940 train + 1470 augmented, 840 validation, {args.epochs} epochs", flush=True)

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "augmentation", "input", "condition", "macro_f1", "n_validation", "parameters"))
        for augmentation, images in training.items():
            train_views = tensor_views(images)
            for seed in args.seeds:
                for mode in ("raw", "clahe", "dual"):
                    torch.manual_seed(seed)
                    batches = torch.Generator().manual_seed(seed)
                    model = SmallCNN(mode == "dual").to(memory_format=torch.channels_last)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
                    model.train()
                    for epoch in range(args.epochs):
                        loss_total = 0.0
                        for idx in torch.randperm(len(train_targets), generator=batches).split(BATCH_SIZE):
                            logits = model(*select_views(train_views, mode, idx))
                            loss = criterion(logits, train_targets[idx])
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                            loss_total += float(loss.detach()) * len(idx)
                        if epoch in (0, args.epochs - 1):
                            print(f"{augmentation} {seed} {mode} epoch {epoch + 1}: "
                                  f"loss {loss_total / len(train_targets):.3f}", flush=True)
                    model.eval()
                    parameters = sum(p.numel() for p in model.parameters())
                    scores = []
                    with torch.inference_mode():
                        for condition, views in validation_views.items():
                            predictions = []
                            for start in range(0, len(valid_y), BATCH_SIZE):
                                idx = slice(start, start + BATCH_SIZE)
                                predicted = model(*select_views(views, mode, idx)).argmax(dim=1)
                                predictions.extend(predicted.tolist())
                            score = f1_score(valid_y, predictions, average="macro")
                            writer.writerow((seed, augmentation, mode, condition, f"{score:.8f}",
                                             len(valid_y), parameters))
                            scores.append(f"{condition}={100 * score:.2f}")
                    handle.flush()
                    print(f"Macro-F1 {augmentation} {seed} {mode}: {' '.join(scores)}", flush=True)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project / "data" / "GTSRB" / "raw")
    parser.add_argument("--output-dir", type=Path, default=project / "results")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs and threads must be positive; seeds must be unique")
    rng = np.random.default_rng(SPLIT_SEED)
    rows = select_split(args.data_root, rng)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = StringIO(newline="")
    writer = csv.writer(manifest)
    writer.writerow(("split", "image", "class_id", "track_id"))
    writer.writerows(rows)
    split_path = args.output_dir / "pilot_split.csv"
    split_bytes = manifest.getvalue().encode("utf-8")
    if not split_path.exists() or split_path.read_bytes() != split_bytes:
        split_path.write_bytes(split_bytes)
    if args.split_only:
        print(f"Wrote the {len(rows)}-row grouped split to {args.output_dir / 'pilot_split.csv'}")
    else:
        run_experiments(args, rows, rng)


if __name__ == "__main__":
    main()
