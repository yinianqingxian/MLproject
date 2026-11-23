"""
用途：定义 A 负责的冻结干扰条件，供开发集十三条件评估与官方测试集评估共用。
核心职责：
- 由已缩放到 32×32 的 BGR 图像生成运动模糊、高斯噪声、变暗和 JPEG 压缩的四族三档强度。
- 开发集复用 `prepare_data` 的干净图和四个 medium 条件像素，保证与原五条件结果直接可比。
- 官方测试集用独立固定种子生成全部十二个干扰条件，三档噪声共享同一标准正态噪声场。
关键副作用/安全约束：
- 只读图像数组：不写文件、不训练、不改变划分、不重排随机序列。
- 参数在查看官方测试结果之前冻结；评估入口不得按测试分数增删或调整强度。
"""

import hashlib

import cv2
import numpy as np

import cnn_robustness_pilot as pilot


# 强度表与 PROTOCOL.md、01-project-plan.md 4.2 节一致，四族各三档加 clean 共十三条件。
LEVELS = {
    "motion": (3, 5, 7),
    "noise": (10, 20, 30),
    "dark": (0.70, 0.45, 0.25),
    "jpeg": (40, 20, 10),
}
SEVERITIES = ("low", "medium", "high")
CONDITION_COUNT = 1 + sum(len(values) for values in LEVELS.values())

# 开发集条件指纹的固定种子（2026-11-17 冻结）。
NOISE_SEED = 20261117
# 官方测试集条件的固定种子，取自 01-project-plan.md 4.2 节。
TEST_NOISE_SEED = 20261111


def _noise_seed(base_seed, image):
    """Derive a stable per-file seed so low/medium/high share one standard normal field."""
    key = f"{base_seed}:{image}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _check_image(image, image_name):
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != (32, 32, 3):
        raise ValueError(f"Conditions require 32x32 uint8 BGR images: {image_name}")
    return array


def _generate(kind, value, clean, rows, noise_seed):
    """Apply one frozen interference at one strength to every clean image."""
    if kind not in LEVELS or value not in LEVELS[kind]:
        raise ValueError(f"Unknown frozen condition: {kind}={value}")
    if len(clean) != len(rows):
        raise ValueError("Each clean image needs one metadata row")
    images = []
    for index, (image, row) in enumerate(zip(clean, rows)):
        image = _check_image(image, row["image"])
        if kind == "motion":
            # Direction alternates on the frozen row order so every strength shares it.
            result = pilot.motion_blur(cv2, image, int(value), index % 2)
        elif kind == "noise":
            generator = np.random.default_rng(_noise_seed(noise_seed, row["image"]))
            result = pilot.gaussian_noise(image, float(value), generator)
        elif kind == "dark":
            result = pilot.darken(image, float(value))
        else:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(value)])
            if not ok:
                raise RuntimeError(f"JPEG encoding failed for {row['image']}")
            result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        images.append(_check_image(result, row["image"]))
    return images


def conditions(data):
    """Yield the frozen development conditions, reusing `prepare_data` pixels for five."""
    clean = data["validation"]["clean"]
    rows = data["validation_rows"]
    if len(clean) != len(rows):
        raise ValueError("Development clean images and metadata rows disagree")
    yield "clean", "none", None, clean
    for kind, values in LEVELS.items():
        for severity, value in zip(SEVERITIES, values):
            if severity == "medium":
                # Reuse the legacy corruption stream so the five-condition results stay comparable.
                images = data["validation"][kind]
            else:
                images = _generate(kind, value, clean, rows, NOISE_SEED)
            yield kind, severity, value, images


def test_conditions(clean, rows):
    """Yield all thirteen official-test conditions generated fresh with `TEST_NOISE_SEED`."""
    if len(clean) != len(rows):
        raise ValueError("Test clean images and metadata rows disagree")
    yield "clean", "none", None, list(clean)
    for kind, values in LEVELS.items():
        for severity, value in zip(SEVERITIES, values):
            yield kind, severity, value, _generate(kind, value, clean, rows, TEST_NOISE_SEED)


def pixel_hash(images):
    """Stable fingerprint of the exact uint8 pixels handed to the model."""
    array = np.ascontiguousarray(images, dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest()


def manifest(specs):
    """Summarize condition order, parameters, counts and pixel fingerprints for auditing."""
    return [
        {
            "condition": kind,
            "severity": severity,
            "parameter": value,
            "n_images": len(images),
            "pixel_sha256": pixel_hash(images),
        }
        for kind, severity, value, images in specs
    ]
