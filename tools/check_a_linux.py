"""
用途：在正式训练前检查 Linux GPU 环境、官方权重与真实显存占用，避免九次训练中途失败。
核心职责：
- 核对 Python、Torch/Torchvision、CUDA、OpenCV 与官方 ResNet18 权重的 SHA256。
- 用正式 batch 做一次前向、反向和优化器更新，报告峰值显存与实际耗时。
- 把环境、驱动和 pip freeze 写入结果目录，作为跨机器比较的前提记录。
关键副作用/安全约束：
- 只在指定输出目录写环境记录；不训练、不保存权重、不写正式结果目录。
"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))

DEFAULT_WEIGHTS = PROJECT / "assets/weights/resnet18-f37072fd.pth"


def _nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True)
        return result.stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        return [f"nvidia-smi unavailable: {type(error).__name__}: {error}"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/a_linux_environment")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-batch-probe", action="store_true",
                        help="跳过正式 batch 的显存探针（不推荐，探针是防 OOM 的主要手段）")
    args = parser.parse_args(argv)

    if platform.system() != "Linux":
        raise SystemExit(f"This check targets the Linux GPU server, not {platform.system()}.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import cv2
    import torch
    import torchvision

    from resnet_robustness_study import ResNet18Fusion, weights_metadata
    from run_c_study import environment_info
    from run_resnet_study import OFFICIAL_WEIGHTS_SHA256

    report = {
        "schema_version": 1,
        "machine": platform.machine(),
        "nvidia_smi": _nvidia_smi(),
        "weights_sha256_expected": OFFICIAL_WEIGHTS_SHA256,
    }
    if not torch.cuda.is_available():
        raise SystemExit(
            "torch.cuda.is_available() is False. Install the GPU build of Torch/Torchvision "
            "matching this server driver; do not continue with a CPU wheel.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    properties = torch.cuda.get_device_properties(0)
    metadata = weights_metadata(pretrained=True, weights_path=args.weights)
    report.update({
        "environment": environment_info("cuda"),
        "torchvision": torchvision.__version__,
        "weights": metadata,
        "gpu": {
            "name": properties.name,
            "capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "total_memory_gib": round(properties.total_memory / 1024 ** 3, 2),
        },
        "cudnn_version": torch.backends.cudnn.version(),
        "opencv": cv2.__version__,
    })
    print(f"GPU: {properties.name} ({report['gpu']['total_memory_gib']} GiB), "
          f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}, "
          f"cudnn {report['cudnn_version']}")

    model = ResNet18Fusion("single", pretrained=True, weights_path=args.weights).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    criterion = torch.nn.CrossEntropyLoss()

    # Small probe first: a failure here is an environment problem, not a memory problem.
    probe = torch.rand(2, 3, 32, 32, device="cuda")
    loss = criterion(model(probe), torch.tensor([0, 1], device="cuda"))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    report["probe"] = {"batch_size": 2, "loss": float(loss.detach().cpu()),
                       "parameters": sum(p.numel() for p in model.parameters())}
    print(f"Batch-2 forward/backward/step OK, loss={report['probe']['loss']:.4f}")

    if not args.skip_batch_probe:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            batch = torch.rand(args.batch_size, 3, 32, 32, device="cuda")
            targets = torch.randint(0, 7, (args.batch_size,), device="cuda")
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        except torch.cuda.OutOfMemoryError as error:
            raise SystemExit(
                f"Batch {args.batch_size} does not fit in {report['gpu']['total_memory_gib']} GiB. "
                "Do not lower the batch size or enable mixed precision locally: the protocol must "
                "stay identical across members. Use a larger GPU or change the protocol with the "
                f"whole group first. Original error: {error}") from error
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        report["batch_probe"] = {
            "batch_size": args.batch_size, "train_step_seconds": elapsed,
            "peak_allocated_bytes": peak, "peak_reserved_bytes": reserved,
            "peak_allocated_gib": round(peak / 1024 ** 3, 2),
            "headroom_reserved_ratio": round(reserved / properties.total_memory, 3),
            "estimated_run_minutes": (4410 / args.batch_size + 0.001) * 10 * elapsed / 60,
            "note": ("One real AdamW step on 32x32 inputs, so the official 224 preprocessing, "
                     "activations and gradients are all included. Not a formal training result."),
        }
        print(f"Batch-{args.batch_size} step: {elapsed:.3f}s, peak {report['batch_probe']['peak_allocated_gib']} GiB "
              f"({report['batch_probe']['headroom_reserved_ratio']:.0%} of device), "
              f"estimated ~{report['batch_probe']['estimated_run_minutes']:.1f} min per run")

    args.output.mkdir(parents=True, exist_ok=True)
    report["status"] = "passed_gpu_environment_and_memory_probe_not_formal_training"
    report["pip_freeze"] = "pip-freeze.txt"
    (args.output / "environment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=True).stdout
    (args.output / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
    print(f"Wrote {args.output / 'environment.json'} and pip-freeze.txt")
    print("Next: bash A_LINUX.sh check")


if __name__ == "__main__":
    main()
