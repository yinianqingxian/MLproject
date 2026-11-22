"""
用途：防止 DGX Spark 的训练入口接受错误架构或没有 GPU 的环境。
核心职责：
- 接受 GB10 的 ARM64 CUDA 运行描述。
- 拒绝 x86、CPU-only 和不匹配的 GPU 计算能力。
"""

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))


class SparkEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("check_spark_environment"),
                             "Spark 环境检查模块尚未实现")
        self.checker = importlib.import_module("check_spark_environment")
        self.runtime = {"platform": "linux", "machine": "aarch64", "cuda_available": True,
                        "gpu_name": "NVIDIA GB10", "compute_capability": [12, 1]}

    def test_accepts_spark_without_assuming_discrete_gpu_memory_reporting(self):
        self.checker.validate_runtime(self.runtime)

    def test_rejects_wrong_architecture_cpu_only_and_wrong_gpu(self):
        for change in ({"machine": "x86_64"}, {"platform": "win32"},
                       {"cuda_available": False}, {"compute_capability": [9, 0]},
                       {"gpu_name": "NVIDIA H100", "compute_capability": [9, 0]}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.checker.validate_runtime(dict(self.runtime, **change))


if __name__ == "__main__":
    unittest.main()
