"""
用途：防止 Spark 恢复训练时混用不同镜像或覆盖已有研究目录。
核心职责：
- 检查首次保存运行环境、相同环境恢复与仅有环境记录时的恢复。
- 拒绝镜像变动、缺失环境记录和未经恢复参数使用已有目录。
关键副作用/安全约束：
- 只在临时目录写入测试环境记录。
"""

import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))


class SparkResumeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("run_spark_study"), "Spark 环境恢复保护尚未实现")
        self.wrapper = importlib.import_module("run_spark_study")
        self.context = {"schema_version": 1, "image_id": "sha256:" + "a" * 64,
                        "torch": "NVIDIA-test", "numpy": "test"}

    def test_new_directory_records_context_and_profile_only_attempt_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "study"
            self.assertTrue(self.wrapper.prepare_study_directory(directory, self.context, resume=False))
            profile = directory / "spark_runtime.json"
            self.assertTrue(profile.is_file())
            before = profile.read_bytes()
            self.assertFalse(self.wrapper.prepare_study_directory(directory, self.context, resume=True))
            self.assertEqual(profile.read_bytes(), before)

    def test_changed_image_and_existing_unprotected_directory_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "study"
            self.wrapper.prepare_study_directory(directory, self.context, resume=False)
            before = (directory / "spark_runtime.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self.wrapper.prepare_study_directory(directory, self.context, resume=False)
            with self.assertRaises(ValueError):
                self.wrapper.prepare_study_directory(directory, dict(self.context, image_id="sha256:" + "b" * 64),
                                                      resume=True)
            self.assertEqual((directory / "spark_runtime.json").read_bytes(), before)
            other = Path(temporary) / "unprotected"
            other.mkdir()
            with self.assertRaises(ValueError):
                self.wrapper.prepare_study_directory(other, self.context, resume=True)
            self.assertEqual(list(other.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
