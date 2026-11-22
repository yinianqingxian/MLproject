"""
用途：核验 Linux 训练环境、固定轨迹划分与 CPU/CUDA 前向及反向计算。
核心职责：
- 验证 Linux 平台、依赖版本、数据数量和预实验划分。
- 输出实际环境、依赖、设备与数据指纹记录。
关键副作用/安全约束：
- 仅向显式指定的报告路径写入 JSON，不训练模型或修改数据与正式结果。
"""

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import struct
import sys
import sysconfig


def check_platform():
    if sys.platform != "linux" or platform.machine() != "x86_64" or struct.calcsize("P") != 8:
        raise RuntimeError("This configuration supports Linux x86_64 only.")
    if sys.implementation.name != "cpython" or sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise RuntimeError("Use CPython 3.12 or 3.13.")
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("Use a regular CPython build, not a free-threaded build.")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
        raise RuntimeError("The pinned PyTorch wheels require glibc >= 2.28.")


def check_packages(root, device):
    expected = {}
    for name in ("requirements-linux.txt", f"requirements-linux-torch-{'cpu' if device == 'cpu' else 'cu132'}.txt"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            package, version = line.split("==", 1)
            actual = metadata.version(package)
            if actual != version:
                raise RuntimeError(f"Dependency mismatch: {package} expected {version}, got {actual}")
            expected[package] = actual
    for package in ("opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"):
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"Remove {package}; install only opencv-python-headless in .venv-linux.")
    return expected


def check_dataset(root, np, cv2):
    data = root / "data/GTSRB/raw"
    train = data / "GTSRB/Final_Training/Images"
    test = data / "GTSRB/Final_Test/Images"
    counts = {"training_images": sum(1 for _ in train.glob("*/*.ppm")),
              "test_images": sum(1 for _ in test.glob("*.ppm"))}
    if counts != {"training_images": 39209, "test_images": 12630}:
        raise RuntimeError(f"Incomplete dataset: {counts}")
    split = root / "results/pilot_split.csv"
    with split.open(encoding="utf-8", newline="") as handle:
        frozen = [(row["split"], row["image"], int(row["class_id"]), row["track_id"])
                  for row in csv.DictReader(handle)]
    sys.path.insert(0, str(root / "experiments"))
    from cnn_robustness_pilot import select_split
    if select_split(data, np.random.default_rng(42)) != frozen:
        raise RuntimeError("Frozen pilot split differs from the dataset; do not regenerate it.")
    sample = cv2.imdecode(np.fromfile(data / frozen[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError("OpenCV could not decode the first frozen training image.")
    return counts, frozen, cv2.resize(sample, (32, 32), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    check_platform()
    root = args.project_root.resolve()
    packages = check_packages(root, args.device)
    import cv2
    import matplotlib
    import numpy as np
    import scipy
    import sklearn
    import torch

    matplotlib.use("Agg")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    counts, frozen, sample = check_dataset(root, np, cv2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check nvidia-smi and the GPU/driver, or install CPU wheels.")
    tensor = torch.from_numpy(sample.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(args.device)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 12, 3, padding=1), torch.nn.GroupNorm(4, 12)).to(args.device)
    result = model(tensor)
    result.square().mean().backward()
    if tuple(result.shape) != (1, 12, 32, 32) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("PyTorch forward check failed.")
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise RuntimeError("PyTorch backward check failed.")
    if args.device == "cuda":
        torch.cuda.synchronize()
    report = {
        "status": "environment_and_data_ready",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "packages": packages, "dataset": counts,
        "split_rows": len(frozen),
        "split_sha256": hashlib.sha256((root / "results/pilot_split.csv").read_bytes()).hexdigest(),
        "device": args.device, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "gpu_capability": torch.cuda.get_device_capability(0) if args.device == "cuda" else None,
        "torch_cuda_arch_list": torch.cuda.get_arch_list() if args.device == "cuda" else [],
        "smoke_check": "32x32 convolution/groupnorm forward and backward passed",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
