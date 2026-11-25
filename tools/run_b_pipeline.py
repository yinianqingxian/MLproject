"""
用途：在 B 的新机器上按顺序完成 12 次训练前的检查、对齐、训练、评估与打包。
核心职责：
- check：环境、依赖版本、数据、划分、权重、代码版本一次看清。
- probe：用真实 batch 128 测单步耗时，估算 12 次训练需要多久（不跑正式训练）。
- inputs：生成开发集条件清单并与 A 的基准比对，13/13 才放行。
- train：clahe 9 次 + denoised_clahe 3 次，完成后只读核验。
- eval：两批各跑十三条件评估，并核对产物数量。
- compare：训练像素 8/8、评估条件 13/13 分别比对。
- package：把训练目录、评估结果、比对报告打成一个压缩包。
关键副作用/安全约束：
- 每一步都会在失败时停下，不带着不一致的输入继续。
- 不覆盖已有目录：训练用 --resume，其余步骤自动改用 _v2、_v3。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
# The probe imports the shared model directly; the other steps shell out to scripts.
sys.path.insert(0, str(PROJECT / "experiments"))

DATA = PROJECT / "data/GTSRB/raw"
SPLIT = PROJECT / "results/pilot_split.csv"
WEIGHTS = PROJECT / "assets/weights/resnet18-f37072fd.pth"
REFERENCE_EVAL = "results/a_eval13_local_v1"
REFERENCE_STUDY = "results/a_local_v1"

CLAHE_STUDY = "results/b_clahe"
DENOISED_STUDY = "results/b_denoised_clahe"
CLAHE_EVAL = "results/b_clahe_eval13"
DENOISED_EVAL = "results/b_denoised_clahe_eval13"
INPUT_CHECK = "results/b_input_check"

SPLIT_SHA = "de914a9199138474115a14776a61574651653ed87c72a26b9804bb35c5aa5359"
WEIGHTS_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
GT_SHA = "58a660fab5465cce66d7334eb9422ae09ed16186ad4500766b59b5a077a1c12f"

TARGET_VERSIONS = {"cv2": "5.0.0", "numpy": "2.4.1"}


def run(command, check=True):
    command = [str(part) for part in command]
    print("$ " + " ".join(command), flush=True)
    process = subprocess.run(command, cwd=str(PROJECT))
    if check and process.returncode != 0:
        raise SystemExit(f"命令失败（exit {process.returncode}）：{' '.join(command)}")
    return process


def free_name(base):
    """Return base, or base_v2 / _v3 ... if it already exists."""
    path = PROJECT / base
    if not path.exists():
        return base
    index = 2
    while (PROJECT / f"{base}_v{index}").exists():
        index += 1
    return f"{base}_v{index}"


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70, flush=True)


def step_check(args):
    banner("1/6 环境与文件检查")
    import platform

    print("platform :", platform.platform(), platform.machine())
    print("python   :", sys.version.split()[0])

    import cv2
    import numpy
    import torch

    print("torch    :", torch.__version__, "| CUDA", torch.version.cuda,
          "| available", torch.cuda.is_available())
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print("GPU      :", properties.name,
              "| %.1f GiB" % (properties.total_memory / 1024 ** 3))
    else:
        print("GPU      : 不可用")
    print("device   :", args.device,
          "（CPU 也能跑，只是慢；一次训练约 350 步 × 单步耗时）" if args.device == "cpu" else "")
    print("cv2      :", cv2.__version__, "| numpy:", numpy.__version__)
    if platform.system() == "Darwin":
        print("核数     :", os.cpu_count(), "| Apple Silicon 请用 --device cpu，"
              "MPS 不在公共训练入口的支持范围内")

    problems = []
    for name, expected in TARGET_VERSIONS.items():
        actual = cv2.__version__ if name == "cv2" else numpy.__version__
        if actual != expected:
            problems.append(f"{name} 版本是 {actual}，基准要求 {expected}")

    if not DATA.is_dir():
        problems.append(f"找不到数据目录 {DATA}")
    else:
        train_dir = DATA / "GTSRB/Final_Training/Images"
        test_dir = DATA / "GTSRB/Final_Test/Images"
        train_count = sum(1 for folder in train_dir.iterdir() if folder.is_dir()
                          for _ in folder.glob("*.ppm"))
        test_count = sum(1 for _ in test_dir.glob("*.ppm"))
        print("训练图   :", train_count, "| 测试图:", test_count)
        if (train_count, test_count) != (39209, 12630):
            problems.append(f"图片数量是 {train_count}/{test_count}，应为 39209/12630")
        gt = DATA / "GT-final_test.csv"
        if not gt.is_file() or sha256(gt) != GT_SHA:
            problems.append("GT-final_test.csv 缺失或指纹不同")

    if not SPLIT.is_file() or sha256(SPLIT) != SPLIT_SHA:
        problems.append("results/pilot_split.csv 缺失或指纹不同")
    else:
        print("划分指纹 :", "一致")
    if not WEIGHTS.is_file() or sha256(WEIGHTS) != WEIGHTS_SHA:
        problems.append("assets/weights/resnet18-f37072fd.pth 缺失或指纹不同（见 assets/weights/README.md）")
    else:
        print("权重指纹 :", "一致")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT),
                            capture_output=True, text=True)
    if commit.returncode == 0:
        print("代码版本 :", commit.stdout.strip())

    print()
    if problems:
        print("检查未通过：")
        for item in problems:
            print("  - " + item)
        raise SystemExit(1)
    print("检查通过：可以进入下一步（inputs 对齐）。")


def step_probe(args):
    banner("2/6 耗时探针（不跑正式训练）")
    import torch

    from resnet_robustness_study import ResNet18Fusion

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda，但当前环境没有可用 CUDA。Mac 请用 --device cpu。")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = ResNet18Fusion("single", pretrained=True, weights_path=WEIGHTS)
    model = model.to(device=device, memory_format=torch.channels_last)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    criterion = torch.nn.CrossEntropyLoss()
    images = torch.rand(128, 3, 32, 32).to(device=device, memory_format=torch.channels_last)
    targets = torch.randint(0, 7, (128,), device=device)

    def one_step():
        loss = criterion(model(images), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # CPU 的每一步要几秒，减少重复次数，避免探针本身花掉几分钟。
    warmup, repeats = (5, 10) if device == "cuda" else (1, 3)
    for _ in range(warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        one_step()
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    samples.sort()
    step_seconds = samples[len(samples) // 2]

    # 4,410 条 / batch 128 = 每轮 35 步；10 轮合计 350 次优化更新（见 cost.json 的 optimizer_steps）。
    steps_per_epoch = 35
    total_updates = 350
    per_epoch = step_seconds * steps_per_epoch
    per_run = step_seconds * total_updates
    eval_seconds = 12 * 13 * (840 / 128 * (step_seconds / 2.6) + 1.5)
    print(f"单步（batch 128）：{step_seconds:.3f} s  [{samples[0]:.3f}–{samples[-1]:.3f}]")
    print(f"一轮（35 步）     ：{per_epoch / 60:.1f} 分钟")
    print(f"一次训练（350 步）：{per_run / 60:.1f} 分钟")
    print(f"12 次训练         ：{12 * per_run / 3600:.1f} 小时")
    print(f"两批十三条件评估  ：约 {eval_seconds / 60:.0f} 分钟")
    print()
    print("注：一轮是 35 步（4,410 条 ÷ batch 128），十轮合计 350 次更新，不是每轮 350 步。")
    print("    笔记本 GPU 起步可能在低频，探针已做 5 步预热；持续满载后通常略快于这个估计。")


def step_inputs(args):
    banner("3/6 开发集条件对齐（不需要 GPU）")
    output = free_name(INPUT_CHECK)
    run([PYTHON, "experiments/evaluate_a13.py", "--prepare-only",
         "--device", "cpu", "--output", output])
    report = "results/b_input_diff.json"
    process = run([PYTHON, "tools/compare_inputs.py", "--reference", REFERENCE_EVAL,
                   "--candidate", output, "--json", report], check=False)
    if process.returncode == 0:
        print("\n13/13 一致，可以开始训练。")
        return
    if not args.allow_pixel_mismatch:
        raise SystemExit(
            "输入未对齐，先解决上面的差异再训练。\n"
            "如果这台机器和 A 不是同一平台（例如 Apple Silicon），像素可能无法逐像素对齐；\n"
            "那种情况用 --allow-pixel-mismatch 继续：组内比较仍然有效，但跨成员差值要改由官方测试统一评估。")
    print("\n警告：与 A 的基准不一致，按 --allow-pixel-mismatch 继续。")
    print("     这不会影响同一台机器内部的组内比较（例如去噪 vs 不去噪）。")
    print("     但它意味着：开发集上 A 与 B 的差值混入了环境因素，不能当作 CLAHE 的效应。")


def step_train(args):
    banner("4/6 正式训练：clahe 9 次 + denoised_clahe 3 次")
    plans = [
        (CLAHE_STUDY, ["--inputs", "clahe", "--augmentations",
                       "repeat_clean", "targeted_motion", "mixed"]),
        (DENOISED_STUDY, ["--inputs", "denoised_clahe", "--augmentations", "mixed"]),
    ]
    for study, extra in plans:
        command = [PYTHON, "experiments/run_resnet_study.py", *extra,
                   "--seeds", "42", "43", "44", "--epochs", "10",
                   "--threads", str(args.threads),
                   "--device", args.device, "--output-dir", study]
        if (PROJECT / study).exists():
            print(f"{study} 已存在，使用 --resume 只补未完成的运行")
            command.append("--resume")
        run(command)
        run(command + ["--verify-only"])


def step_eval(args):
    banner("5/6 十三条件评估")
    for study, output_slug, expected in ((CLAHE_STUDY, CLAHE_EVAL, 9),
                                         (DENOISED_STUDY, DENOISED_EVAL, 3)):
        if not (PROJECT / study / "run_config.json").is_file():
            raise SystemExit(f"{study} 不存在或不完整，先跑 train。")
        output = free_name(output_slug)
        run([PYTHON, "experiments/evaluate_a13.py", "--study", study, "--output", output,
             "--device", args.device, "--expect-runs", str(expected)])

    rows, predictions = 0, 0
    for slug in (CLAHE_EVAL, DENOISED_EVAL):
        base = PROJECT / slug
        if not base.is_dir():
            continue
        for run in base.iterdir():
            if run.is_dir() and (run / "metrics.csv").is_file():
                with (run / "metrics.csv").open(encoding="utf-8") as handle:
                    rows += max(0, sum(1 for _ in handle) - 1)
                with (run / "predictions.csv").open(encoding="utf-8") as handle:
                    predictions += max(0, sum(1 for _ in handle) - 1)
    print(f"\n指标行 {rows}（应为 156）| 逐图预测 {predictions}（应为 131,040）")
    if (rows, predictions) != (156, 131040):
        raise SystemExit("产物数量不对，先别打包。")


def step_compare(args):
    banner("6/6 训练像素与评估条件比对")
    pairs = [
        (REFERENCE_STUDY, CLAHE_STUDY, "results/b_clahe_train_diff.json"),
        (REFERENCE_STUDY, DENOISED_STUDY, "results/b_denoised_train_diff.json"),
        (REFERENCE_EVAL, CLAHE_EVAL, "results/b_clahe_eval_diff.json"),
        (REFERENCE_EVAL, DENOISED_EVAL, "results/b_denoised_eval_diff.json"),
    ]
    failed = 0
    for reference, candidate, output in pairs:
        if not (PROJECT / candidate).exists():
            print(f"跳过 {candidate}（不存在）")
            continue
        process = run([PYTHON, "tools/compare_inputs.py", "--reference", reference,
                       "--candidate", candidate, "--json", output], check=False)
        failed += 0 if process.returncode == 0 else 1
    print()
    if failed:
        print(f"{failed} 项比对未通过；产物仍可交给 A 判断，但结论要按实际差异写。")
    else:
        print("全部一致。")


def step_package(args):
    banner("打包")
    archive = PROJECT / "b_newpc_results.zip"
    if archive.exists():
        archive = PROJECT / "b_newpc_results_v2.zip"
    included = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
        for slug in (CLAHE_STUDY, DENOISED_STUDY, CLAHE_EVAL, DENOISED_EVAL,
                     INPUT_CHECK, "results"):
            base = PROJECT / slug
            if not base.exists():
                continue
            paths = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
            for path in paths:
                relative = path.relative_to(PROJECT).as_posix()
                if slug == "results" and not relative.endswith("_diff.json"):
                    continue
                handle.write(path, relative)
                included += 1
    print(f"{archive.name}：{included} 个文件，{archive.stat().st_size / 1024 ** 2:.1f} MB")


STEPS = {"check": step_check, "probe": step_probe, "inputs": step_inputs,
         "train": step_train, "eval": step_eval, "compare": step_compare,
         "package": step_package}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=list(STEPS) + ["all"],
                        help="check → probe → inputs → train → eval → compare → package")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto 时优先 CUDA；Apple Silicon 用 cpu（MPS 不在公共训练入口支持范围）")
    parser.add_argument("--threads", type=int, default=6, help="CPU 线程数；Mac 上可以调到 8~12 提速")
    parser.add_argument("--allow-pixel-mismatch", action="store_true",
                        help="开发集条件与 A 不一致时仍继续（Mac/跨平台）；差异会写入 b_input_diff.json")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须是正整数")
    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    if args.step == "all":
        for name in ("check", "probe", "inputs", "train", "eval", "compare", "package"):
            STEPS[name](args)
    else:
        STEPS[args.step](args)


if __name__ == "__main__":
    main()
