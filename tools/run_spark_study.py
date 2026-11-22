"""
用途：在保持公共训练源码冻结的同时保护 Spark 实验的容器环境一致性。
核心职责：
- 在训练前保存不可变镜像标识、依赖与驱动环境。
- 拒绝不同环境恢复，再转交公共 CNN 训练入口。
关键副作用/安全约束：
- 仅为新的研究目录预留环境记录；已有目录须显式恢复并通过环境核验。
"""

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]


def prepare_study_directory(directory, runtime, resume):
    """Reserve a new study or require an exact match before resuming an old one."""
    directory = Path(directory)
    profile = directory / "spark_runtime.json"
    if directory.exists():
        if not resume:
            raise FileExistsError(f"Spark output exists; choose a new directory or --resume: {directory}")
        try:
            previous = json.loads(profile.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("Spark runtime record missing/invalid; preserve this directory and use a new one") from error
        if previous != runtime:
            raise ValueError("Spark image, driver, dependencies or runtime flags changed; use a new output directory")
        return False
    directory.mkdir(parents=True, exist_ok=False)
    with profile.open("x", encoding="utf-8") as handle:
        json.dump(runtime, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return True


def runtime_profile():
    from check_spark_environment import validate_runtime
    import cv2
    import numpy as np
    import scipy
    import torch
    import torchvision
    image = os.environ.get("SPARK_CONTAINER_IMAGE_ID", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise RuntimeError("Immutable image ID missing; start training with bash SPARK.sh train")
    target = {"platform": sys.platform, "machine": platform.machine(),
              "cuda_available": bool(torch.cuda.is_available())}
    if target["cuda_available"]:
        target.update(gpu_name=torch.cuda.get_device_name(0),
                      compute_capability=list(torch.cuda.get_device_capability(0)))
    validate_runtime(target)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout.strip()
    return {
        "schema_version": 1, "image_id": image, "target": target,
        "python": platform.python_version(), "platform": platform.platform(), "driver": driver,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "protected_core": {name: {"version": str(module.__version__), "path": str(Path(module.__file__).resolve())}
                           for name, module in (("torch", torch), ("torchvision", torchvision), ("numpy", np), ("scipy", scipy))},
        "opencv": cv2.__version__, "opencv_path": str(Path(cv2.__file__).resolve()),
        "application_versions": {name: metadata.version(name) for name in ("scikit-learn", "matplotlib")},
        "runtime_flags": {name: os.environ.get(name) for name in (
            "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "NVIDIA_IMEX_CHANNELS")},
    }


def main(argv=None):
    forwarded = list(sys.argv[1:] if argv is None else argv)
    sys.path.insert(0, str(PROJECT / "experiments"))
    from run_resnet_study import main as train_main
    if "--help" in forwarded or "-h" in forwarded:
        return train_main(forwarded)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/study_resnet18_spark_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    options, _ = parser.parse_known_args(forwarded)
    # Keep the wrapper's reserved directory and GPU default identical to what
    # the core receives, including direct invocation without launcher defaults.
    forwarded.extend(["--output-dir", str(options.output_dir), "--device", options.device])
    if options.verify_only:
        return train_main(forwarded)
    runtime = runtime_profile()
    created = prepare_study_directory(options.output_dir, runtime, options.resume)
    if created:
        # The wrapper created only the runtime record. Let the unchanged core
        # create its run_config and new runs within this newly reserved directory.
        forwarded.append("--resume")
    return train_main(forwarded)


if __name__ == "__main__":
    main()
