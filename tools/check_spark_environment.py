"""
用途：核验 DGX Spark 容器能按既定协议运行交通标志 CNN。
核心职责：
- 检查 ARM64、GB10、CUDA、固定数据划分与科学计算依赖。
- 执行六种输入模式的真实 GPU 优化步骤，记录容器及环境证据。
关键副作用/安全约束：
- 只在显式指定路径写入环境报告，不保存模型或修改正式研究结果。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def validate_runtime(runtime):
    """Accept the GB10 ARM64 CUDA target without relying on UMA memory counters."""
    if runtime.get("platform") != "linux" or runtime.get("machine", "").lower() not in (
            "aarch64", "arm64"):
        raise RuntimeError("DGX Spark requires Linux ARM64; the x86_64 launcher is separate.")
    if not runtime.get("cuda_available"):
        raise RuntimeError("CUDA unavailable: use the NVIDIA Spark image with Docker --gpus all.")
    if tuple(runtime.get("compute_capability") or ()) != (12, 1):
        raise RuntimeError("Expected Spark GB10 compute capability 12.1; check the selected GPU/image.")
    if "GB10" not in runtime.get("gpu_name", "").upper():
        raise RuntimeError("Expected the DGX Spark GB10 GPU.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if sys.platform != "linux" or platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("Run this check inside the Spark ARM64 container: bash SPARK.sh check")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The pinned NVIDIA 25.11 profile requires its Python 3.12 environment.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import cv2
    import matplotlib
    import numpy as np
    import scipy
    import sklearn
    from sklearn.metrics import f1_score
    import torch
    import torchvision

    runtime = {"platform": sys.platform, "machine": platform.machine(),
               "cuda_available": bool(torch.cuda.is_available())}
    if runtime["cuda_available"]:
        runtime["gpu_name"] = torch.cuda.get_device_name(0)
        runtime["compute_capability"] = list(torch.cuda.get_device_capability(0))
    validate_runtime(runtime)
    constraint = Path("/etc/pip/constraint.txt")
    if not constraint.is_file():
        raise RuntimeError("NVIDIA base constraints missing; rebuild the pinned Spark image.")

    root = args.project_root.resolve()
    # The generic helper's data routine is portable; its x86-only main is not invoked.
    from check_linux_environment import check_dataset
    cv2.setNumThreads(1)
    counts, rows, sample = check_dataset(root, np, cv2)
    if f1_score([1, 2, 3, 4, 5, 7, 8], [1, 2, 3, 4, 5, 7, 8],
                labels=[1, 2, 3, 4, 5, 7, 8], average="macro", zero_division=0) != 1.0:
        raise RuntimeError("scikit-learn metric check failed")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    buffer = BytesIO()
    figure, axis = plt.subplots(figsize=(2, 2))
    axis.plot([0, 1], [0, 1])
    figure.savefig(buffer, format="png")
    plt.close(figure)
    if not buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Matplotlib Agg PNG check failed")

    sys.path.insert(0, str(root / "experiments"))
    from resnet_robustness_study import INPUT_MODES, ResNet18Fusion, tensor_views, weights_metadata
    pretrained_metadata = weights_metadata()
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    views = tensor_views([sample.copy() for _ in range(4)])
    targets = torch.tensor([0, 1, 2, 3], device="cuda")
    probes = []
    for mode, (fusion, names) in INPUT_MODES.items():
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        model = ResNet18Fusion(fusion).to("cuda", memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        inputs = [views[name].to("cuda", memory_format=torch.channels_last) for name in names]
        logits = model(*inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if logits.dtype != torch.float32 or tuple(logits.shape) != (4, 7) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"GPU FP32 forward failed: {mode}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise RuntimeError(f"GPU backward failed: {mode}")
        optimizer.step()
        torch.cuda.synchronize()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError(f"GPU AdamW step failed: {mode}")
        probes.append({"input": mode, "probe_loss": float(loss.detach().cpu()),
                       "status": "fp32_forward_backward_adamw_passed"})

    core = {name: {"version": str(module.__version__), "module_file": str(module.__file__)}
            for name, module in (("torch", torch), ("torchvision", torchvision), ("numpy", np), ("scipy", scipy))}
    drivers = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15, check=True,
    ).stdout.strip()
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and name not in packages:
            packages[name] = metadata.version(name)
    report = {
        "status": "spark_environment_and_data_ready", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime, "python": sys.version, "platform": platform.platform(),
        "container_image_ref": os.environ.get("SPARK_CONTAINER_IMAGE_REF", "not_provided"),
        "container_image_id": os.environ.get("SPARK_CONTAINER_IMAGE_ID", "not_provided"),
        "protected_core": core, "cuda_runtime": torch.version.cuda,
        "model": pretrained_metadata,
        "cudnn": torch.backends.cudnn.version(), "driver": drivers,
        "opencv": cv2.__version__, "opencv_module_file": cv2.__file__,
        "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "nvidia_constraint_sha256": hashlib.sha256(constraint.read_bytes()).hexdigest(),
        "environment_flags": {name: os.environ.get(name) for name in (
            "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "NVIDIA_IMEX_CHANNELS", "PIP_CONSTRAINT")},
        "dataset": counts, "split_rows": len(rows),
        "split_sha256": hashlib.sha256((root / "results/pilot_split.csv").read_bytes()).hexdigest(),
        "gpu_probes": probes, "metric_and_png_checks": "passed",
        "installed_packages": dict(sorted(packages.items())),
        "scope": "readiness probes only; no formal study training or official-test evaluation",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Keep terminal output small; the report contains the complete environment.
    print(json.dumps({key: report[key] for key in (
        "status", "runtime", "container_image_ref", "container_image_id", "protected_core",
        "cuda_runtime", "driver", "dataset", "split_rows", "gpu_probes")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
