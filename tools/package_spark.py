"""
用途：生成排除 Windows 环境的独立 DGX Spark 训练 ZIP。
核心职责：
- 选取完整原始数据、冻结划分和必需源码，生成 Spark 专用说明与清单。
- 核对原始数据指纹、压缩包成员及文件校验和。
关键副作用/安全约束：
- 只创建新的 ZIP、校验和与核验记录，拒绝覆盖已有交付物。
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT_NAME = "DS4023_TrafficSign"
PROJECT = Path(__file__).resolve().parents[1]


def spark_summary_documentation(payload):
    """Adapt only generated README instructions; retain all numerical code."""
    text = payload.decode("utf-8")
    start = text.index("def build_readme(")
    end = text.index("\ndef source_manifest(", start)
    section = text[start:end]
    slash = chr(92)
    windows_prefix = "." + slash * 2 + ".venv" + slash * 2 + "Scripts" + slash * 2 + "python.exe ." + slash * 2 + "experiments" + slash * 2
    if section.count(windows_prefix) != 3:
        raise ValueError("Unexpected report README template; inspect before packaging")
    section = section.replace(windows_prefix, "python experiments/")
    section = section.replace("f'python experiments/run_c_study.py --output-dir", "f'python tools/run_spark_study.py --output-dir", 1)
    section = section.replace("```powershell", "```bash")
    section = section.replace("在项目根目录使用本项目 Python 环境运行：", "以下 Python 命令在 Spark 容器内的 /workspace 执行；宿主机优先使用 bash SPARK.sh 的相应入口：")
    section = section.replace("固定版本见项目 `requirements.lock`。", "实际版本见本次研究的 `spark_runtime.json` 和 Spark 环境报告。")
    section = section.replace("协议来源为项目 HANDOFF_C.md、02-c-implementation.md 和 01-project-plan.md 的成员 C 与固定协议章节。", "本训练包协议见项目 PROTOCOL.md 与 DGX_SPARK.md。")
    packaged = text[:start] + section + text[end:]
    before, after = ast.parse(text), ast.parse(packaged)
    for tree in (before, after):
        tree.body = [item for item in tree.body if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or item.name != "build_readme"]
    if ast.dump(before) != ast.dump(after):
        raise ValueError("Report adaptation changed code outside README generation")
    return packaged.encode("utf-8")


def package(project, output, workers=12):
    project, output = Path(project).resolve(), Path(output).resolve()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    verified_path = output.with_suffix(output.suffix + ".verification.json")
    if any(path.exists() for path in (output, checksum_path, verified_path)):
        raise FileExistsError("Choose a new ZIP filename; delivery files are never overwritten")
    original = json.loads((project / "PACKAGE_MANIFEST.transfer_original.json").read_text(encoding="utf-8"))
    data = [record for record in original["files"] if record["path"].startswith("data/GTSRB/raw/")]
    counts = {"training_images": sum(record["path"].endswith(".ppm") and "/Final_Training/Images/" in record["path"] for record in data),
              "test_images": sum(record["path"].endswith(".ppm") and "/Final_Test/Images/" in record["path"] for record in data)}
    if counts != {"training_images": 39209, "test_images": 12630} or len(data) != 51886:
        raise ValueError(f"Original complete dataset manifest differs: {counts}, files={len(data)}")
    selected = ["SPARK.sh", "03-resnet-c-implementation.md", "results/pilot_split.csv", "tools/check_linux_environment.py",
                "tools/check_spark_environment.py", "tools/run_spark_study.py", "tools/verify_spark_package.py",
                "tools/audit_b_handoff.py"]
    for folder in (project / "experiments", project / "spark"):
        selected.extend(path.relative_to(project).as_posix() for path in folder.rglob("*")
                        if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".txt", ".sh", "") )
    selected = sorted(set(selected))
    custom = {"START_HERE.md": (project / "tools/spark_package_START.md").read_bytes(),
              "PROTOCOL.md": (project / "tools/spark_package_PROTOCOL.md").read_bytes(),
              "experiments/summarize_c_study.py": spark_summary_documentation((project / "experiments/summarize_c_study.py").read_bytes())}
    guide = (project / "DGX_SPARK.md").read_text(encoding="utf-8")
    copy_start = guide.index("## 1. 搬运和前提")
    copy_end = guide.index("在 Spark 的终端", copy_start)
    guide = guide[:copy_start] + "## 1. 搬运和前提\n\n此独立 ResNet18 训练包已包括 SPARK.sh、spark/、必需的 tools/、experiments/、完整 data/GTSRB/raw/、results/pilot_split.csv 及 assets/weights/ 官方初始化。不含 Windows 环境、旧模型或旧报告；它们保留在原电脑。\n\n" + guide[copy_end:]
    guide = guide.replace("| 原 Windows 结果与报告 | `results/study_v1/`、`report/c_part/` |", "| 原 Windows 结果与报告 | 不随本训练包携带，保留在原工作目录 |")
    custom["DGX_SPARK.md"] = guide.encode("utf-8")
    for name in custom:
        if name not in selected:
            selected.append(name)
    expected = {record["path"]: record for record in data}
    selected.extend(path.relative_to(project).as_posix() for path in (project / "assets/weights").iterdir()
                    if path.is_file() and path.name in ("resnet18-f37072fd.pth", "README.md"))
    weights_payload = (project / "assets/weights/resnet18-f37072fd.pth").read_bytes()
    weights_sha = hashlib.sha256(weights_payload).hexdigest()
    if weights_sha != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
        raise ValueError("Official ResNet18 initialization checksum mismatch")
    records = data + [{"path": relative} for relative in sorted(set(selected))]
    records.sort(key=lambda record: record["path"])
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("Duplicate source selection")

    def read_record(record):
        relative = record["path"]
        safe = PurePosixPath(relative)
        if safe.is_absolute() or ".." in safe.parts or ":" in relative:
            raise ValueError("Unsafe package path")
        payload = custom[relative] if relative in custom else (project / relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if relative in expected and (len(payload) != record["bytes"] or digest != record["sha256"]):
            raise ValueError("Original dataset changed: " + relative)
        return relative, payload, {"path": relative, "bytes": len(payload), "sha256": digest}

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        def write_member(relative, payload):
            info = zipfile.ZipInfo(ROOT_NAME + "/" + relative, (2026, 11, 17, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if relative.endswith(".sh") else 0o100644) << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(records), 512):
                for relative, payload, metadata in pool.map(read_record, records[start:start + 512]):
                    write_member(relative, payload)
                    manifest_files.append(metadata)
                if start == 0 or (start // 512) % 10 == 0 or start + 512 >= len(records):
                    print(f"Packed and source-hash checked {len(manifest_files)}/{len(records)} files", flush=True)
        manifest = {"format_version": 1, "package_kind": "dgx_spark_training", "archive_root": ROOT_NAME,
                    "created_date": "2026-11-17", "dataset": counts, "dataset_files": len(data),
                    "includes_windows_environment": False, "includes_historical_models": False,
                    "backbone": "torchvision.models.resnet18", "feature_dim": 512,
                    "weights_id": "ResNet18_Weights.IMAGENET1K_V1", "weights_sha256": weights_sha,
                    "requires_network_for_container": True, "spark_physical_execution_tested": False,
                    "documentation_adjustments": "Spark-only START/PROTOCOL/guide; report README examples adapted only, numerical code unchanged",
                    "files": manifest_files}
        write_member("PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    from verify_spark_package import verify_zip
    verification = verify_zip(output, progress=True)
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    with checksum_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest.hexdigest()}  {output.name}\n")
    verification.update(zip=str(output), zip_bytes=output.stat().st_size, zip_sha256=digest.hexdigest(),
                        all_original_raw_data_fingerprints_verified=True)
    with verified_path.open("x", encoding="utf-8") as handle:
        json.dump(verification, handle, ensure_ascii=False, indent=2)
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=PROJECT.parent / "DS4023_TrafficSign_ResNet18_Spark_20261117.zip")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    package(args.project, args.output, args.workers)


if __name__ == "__main__":
    main()
