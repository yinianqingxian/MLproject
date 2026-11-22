"""
用途：使用微型数据核验 DGX Spark 交付包的完整性与路径安全检查。
核心职责：
- 验证目录和 ZIP 的文件清单、数据计数、SHA256 与 CRC 校验。
- 拒绝篡改、重复条目、越界路径和逃逸目录的符号链接。
关键副作用/安全约束：
- 测试只在项目的 .test_tmp 中创建并清理微型临时文件。
"""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))


class SparkPackageTests(unittest.TestCase):
    def setUp(self):
        self.verifier = importlib.import_module("verify_spark_package")
        temporary_root = PROJECT_ROOT / ".test_tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.project = self.base / "DS4023_TrafficSign"
        self.project.mkdir()
        self.payloads = {
            "data/GTSRB/raw/GTSRB/Final_Training/Images/00000/00000_00000.ppm": b"training",
            "data/GTSRB/raw/GTSRB/Final_Test/Images/00000.ppm": b"test",
            "tools/example.py": b"print('tiny fixture')\n",
        }
        self.manifest = {
            "package_kind": "dgx_spark_training",
            "archive_root": "DS4023_TrafficSign",
            "dataset": {"training_images": 1, "test_images": 1},
            "files": [
                {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in self.payloads.items()
            ],
        }
        for name, data in self.payloads.items():
            target = self.project / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        self.write_manifest()

    def write_manifest(self):
        (self.project / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(self.manifest), encoding="utf-8-sig")

    def write_zip(self, *, payloads=None, extra=None):
        target = self.base / "fixture.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("DS4023_TrafficSign/PACKAGE_MANIFEST.json",
                             (self.project / "PACKAGE_MANIFEST.json").read_bytes())
            for name, data in (self.payloads if payloads is None else payloads).items():
                archive.writestr("DS4023_TrafficSign/" + name, data)
            for name, data in (extra or []):
                archive.writestr(name, data)
        return target

    def test_verifies_zip_and_directory_without_counting_generated_results(self):
        generated = self.project / "results" / "new-run.json"
        generated.parent.mkdir()
        generated.write_text("{}", encoding="utf-8")
        for result in (self.verifier.verify_directory(self.project),
                       self.verifier.verify_zip(self.write_zip())):
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["verified_files"], 3)
            self.assertEqual(result["verified_bytes"], sum(map(len, self.payloads.values())))
            self.assertEqual(result["dataset"], {"training_images": 1, "test_images": 1})

    def test_rejects_changed_bytes_in_zip_and_directory(self):
        changed = dict(self.payloads, **{"tools/example.py": b"changed"})
        with self.assertRaisesRegex(ValueError, "(SHA256|size|bytes)"):
            self.verifier.verify_zip(self.write_zip(payloads=changed))
        (self.project / "tools/example.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "(SHA256|size|bytes)"):
            self.verifier.verify_directory(self.project)

    def test_reads_zip_payloads_to_detect_crc_damage(self):
        archive_path = self.write_zip()
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo("DS4023_TrafficSign/" + next(iter(self.payloads)))
            offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        data = bytearray(archive_path.read_bytes())
        data[offset] ^= 1
        archive_path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "(CRC|ZIP)"):
            self.verifier.verify_zip(archive_path)

    def test_rejects_duplicate_zip_members_and_extra_inventory(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive = self.write_zip(extra=[("DS4023_TrafficSign/tools/example.py", b"duplicate")])
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            self.verifier.verify_zip(archive)
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.verifier.verify_zip(self.write_zip(extra=[("DS4023_TrafficSign/extra.txt", b"extra")]))

    def test_rejects_duplicate_records_and_inconsistent_dataset_counts(self):
        self.manifest["files"].append(dict(self.manifest["files"][0]))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            self.verifier.verify_directory(self.project)
        self.manifest["files"].pop()
        self.manifest["dataset"]["training_images"] = 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "training_images"):
            self.verifier.verify_directory(self.project)

    def test_cli_reports_success_progress_and_clear_nonzero_failure(self):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = self.verifier.main([str(self.write_zip()), "--progress"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "passed")
        self.assertIn("Verified 3/3 files", errors.getvalue())
        errors = io.StringIO()
        with redirect_stderr(errors):
            status = self.verifier.main([str(self.base / "missing.zip")])
        self.assertEqual(status, 1)
        self.assertTrue(errors.getvalue().startswith("Error:"))

    def test_rejects_unsafe_manifest_and_zip_paths(self):
        for unsafe in ("../escape", "/absolute", "C:/drive", "tools\\bad.py", "tools/./bad.py"):
            with self.subTest(unsafe=unsafe):
                self.manifest["files"][0]["path"] = unsafe
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "[Uu]nsafe"):
                    self.verifier.verify_directory(self.project)
        self.manifest["files"][0]["path"] = next(iter(self.payloads))
        self.write_manifest()
        for unsafe in ("../escape", "other_root/file", "DS4023_TrafficSign/../escape"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                self.verifier.verify_zip(self.write_zip(extra=[(unsafe, b"bad")]))

    def test_rejects_symlink_escape_when_supported(self):
        external = self.base / "external.py"
        external.write_bytes(self.payloads["tools/example.py"])
        target = self.project / "tools/example.py"
        target.unlink()
        try:
            target.symlink_to(external)
        except OSError as error:
            self.skipTest("Symlinks unavailable: " + str(error))
        with self.assertRaisesRegex(ValueError, "(outside|escape)"):
            self.verifier.verify_directory(self.project)


if __name__ == "__main__":
    unittest.main()
