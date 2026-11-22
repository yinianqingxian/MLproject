"""
用途：仅使用 Python 标准库核验 DGX Spark 训练交付目录或 ZIP 包。
核心职责：
- 检查清单格式、数据图像计数与安全的项目相对路径。
- 逐文件读取并比对字节数和 SHA256，读取 ZIP 时同时触发 CRC 校验。
关键副作用/安全约束：
- 只读取交付内容，不解压、不导入训练依赖、不修改数据或训练结果。
- 拒绝重复条目、清单外 ZIP 文件和逃逸项目目录的符号链接。
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import zipfile
import zlib


PACKAGE_KIND = "dgx_spark_training"
ARCHIVE_ROOT = "DS4023_TrafficSign"
MANIFEST_NAME = "PACKAGE_MANIFEST.json"
CHUNK_BYTES = 1024 * 1024
TRAINING_PREFIX = "data/GTSRB/raw/GTSRB/Final_Training/Images/"
TEST_PREFIX = "data/GTSRB/raw/GTSRB/Final_Test/Images/"


class VerificationError(ValueError):
    """The package cannot be trusted as the declared Spark delivery."""


def _safe_path(value, *, context):
    if (not isinstance(value, str) or not value or "\\" in value
            or ":" in value or any(ord(char) < 32 for char in value)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise VerificationError(f"Unsafe {context} path: {value!r}")
    return value


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"Duplicate manifest JSON key: {key!r}")
        result[key] = value
    return result


def _parse_manifest(data):
    try:
        manifest = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"Invalid {MANIFEST_NAME}: {error}") from error
    if not isinstance(manifest, dict):
        raise VerificationError("Manifest must be a JSON object.")
    if manifest.get("package_kind") != PACKAGE_KIND:
        raise VerificationError(f"Manifest package_kind must be {PACKAGE_KIND!r}.")
    if manifest.get("archive_root") != ARCHIVE_ROOT:
        raise VerificationError(f"Manifest archive_root must be {ARCHIVE_ROOT!r}.")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise VerificationError("Manifest dataset must be an object.")
    for name in ("training_images", "test_images"):
        value = dataset.get(name)
        if type(value) is not int or value < 0:
            raise VerificationError(f"Manifest dataset.{name} must be a nonnegative integer.")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise VerificationError("Manifest files must be an array.")
    seen = set()
    counts = {"training_images": 0, "test_images": 0}
    for record in records:
        if not isinstance(record, dict):
            raise VerificationError("Each manifest file record must be an object.")
        path = _safe_path(record.get("path"), context="manifest")
        if path.casefold() == MANIFEST_NAME.casefold():
            raise VerificationError("Manifest must exclude itself from files.")
        if path.casefold() in seen:
            raise VerificationError(f"Duplicate manifest file path: {path!r}")
        seen.add(path.casefold())
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise VerificationError(f"Invalid bytes for {path!r}: expected a nonnegative integer.")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise VerificationError(f"Invalid SHA256 for {path!r}.")
        if path.endswith(".ppm"):
            if path.startswith(TRAINING_PREFIX):
                counts["training_images"] += 1
            elif path.startswith(TEST_PREFIX):
                counts["test_images"] += 1
    for name, actual in counts.items():
        if dataset[name] != actual:
            raise VerificationError(
                f"Manifest dataset.{name} declares {dataset[name]} but files contain {actual} images.")
    return manifest


def _check_stream(stream, record):
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    if size != record["bytes"]:
        raise VerificationError(
            f"File size mismatch for {record['path']!r}: expected {record['bytes']} bytes, got {size}.")
    if digest.hexdigest() != record["sha256"].lower():
        raise VerificationError(f"SHA256 mismatch for {record['path']!r}.")
    return size


def _report_progress(count, total, byte_count, enabled):
    if enabled and (count % 10000 == 0 or count == total):
        print(f"Verified {count}/{total} files ({byte_count:,} bytes)", file=sys.stderr, flush=True)


def _result(manifest, count, byte_count):
    return {
        "status": "passed",
        "package_kind": manifest["package_kind"],
        "archive_root": manifest["archive_root"],
        "verified_files": count,
        "verified_bytes": byte_count,
        "dataset": dict(manifest["dataset"]),
    }


def _directory_file(root, relative):
    target = root.joinpath(*relative.split("/"))
    try:
        resolved = target.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise VerificationError(f"Cannot resolve declared file {relative!r}: {error}") from error
    if not resolved.is_relative_to(root):
        raise VerificationError(f"Declared file escapes outside project root: {relative!r}")
    if not resolved.is_file():
        raise VerificationError(f"Declared path is not a regular file: {relative!r}")
    return resolved


def verify_directory(project_root, progress=False):
    """Verify declared files; allow undeclared outputs created by later training."""
    try:
        root = Path(project_root).resolve(strict=True)
        if not root.is_dir():
            raise VerificationError(f"Project root is not a directory: {root}")
        manifest = _parse_manifest(_directory_file(root, MANIFEST_NAME).read_bytes())
        count = byte_count = 0
        total = len(manifest["files"])
        for record in manifest["files"]:
            with _directory_file(root, record["path"]).open("rb") as stream:
                byte_count += _check_stream(stream, record)
            count += 1
            _report_progress(count, total, byte_count, progress)
        return _result(manifest, count, byte_count)
    except OSError as error:
        raise VerificationError(f"Cannot verify directory: {error}") from error


def verify_zip(archive_path, progress=False):
    """Verify exact ZIP inventory and every declared byte without extracting."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = {}
            seen = set()
            for info in archive.infolist():
                name = _safe_path(info.orig_filename, context="ZIP member")
                if name.casefold() in seen:
                    raise VerificationError(f"Duplicate ZIP member: {name!r}")
                seen.add(name.casefold())
                if not name.startswith(ARCHIVE_ROOT + "/"):
                    raise VerificationError(f"ZIP member is outside archive root: {name!r}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise VerificationError(f"ZIP symlink is not allowed: {name!r}")
                members[name] = info
            manifest_path = ARCHIVE_ROOT + "/" + MANIFEST_NAME
            if manifest_path not in members:
                raise VerificationError(f"ZIP is missing {manifest_path}.")
            manifest = _parse_manifest(archive.read(members[manifest_path]))
            expected = {ARCHIVE_ROOT + "/" + record["path"] for record in manifest["files"]}
            expected.add(manifest_path)
            actual = set(members)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise VerificationError(
                    f"ZIP inventory mismatch: missing={missing[:5]!r}, extra={extra[:5]!r}.")
            count = byte_count = 0
            total = len(manifest["files"])
            for record in manifest["files"]:
                info = members[ARCHIVE_ROOT + "/" + record["path"]]
                with archive.open(info) as stream:
                    byte_count += _check_stream(stream, record)
                count += 1
                _report_progress(count, total, byte_count, progress)
            return _result(manifest, count, byte_count)
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error) as error:
        raise VerificationError(f"Cannot verify ZIP (including CRC): {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a Spark delivery directory or ZIP using only Python's standard library.")
    parser.add_argument("path", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--progress", action="store_true", help="Print progress every 10,000 files and at completion.")
    args = parser.parse_args(argv)
    try:
        verifier = verify_directory if args.path.is_dir() else verify_zip
        result = verifier(args.path, progress=args.progress)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
