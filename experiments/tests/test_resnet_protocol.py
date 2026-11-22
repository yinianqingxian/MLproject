"""
用途：核验独立 ResNet 入口的科学配置、矩阵与只读恢复边界。
核心职责：
- 拒绝旧骨干、错误预处理和不完整的研究元数据。
- 在无权重文件及无 GPU 依赖时验证完成产物并检测篡改。
关键副作用/安全约束：
- 仅在临时目录创建核验夹具，不启动正式训练或下载权重。
"""

import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ResNetProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("run_resnet_study"),
                             "独立 ResNet 运行入口尚未实现")
        self.runner = importlib.import_module("run_resnet_study")

    def official_common(self):
        # Independently specified protocol, rather than the runner's config builder.
        return {
            "schema_version": 1, "classes": [1, 2, 3, 4, 5, 7, 8], "epochs": 10,
            "batch_size": 128, "learning_rate": 0.003, "optimizer": "AdamW",
            "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
            "image_size": 32, "threads": 6, "device": "cuda", "training_entries": 4410,
            "clean_training_images": 2940, "validation_images": 840,
            "train_tracks": 98, "validation_tracks": 28,
            "data_fingerprint": "a" * 64, "split_sha256": hashlib.sha256(b"split\n").hexdigest(),
            "evaluation": "final_epoch_development_validation",
            "clahe": {"clip_limit": 2.0, "tile_grid": [4, 4], "channel": "LAB_L"},
            "denoising": {"method": "bilateral", "d": 5, "sigmaColor": 25, "sigmaSpace": 2},
            "backbone": "torchvision.models.resnet18", "feature_dim": 512,
            "network_input_size": 224, "pretrained": True,
            "weights_id": "ResNet18_Weights.IMAGENET1K_V1",
            "weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            "weights_path": "assets/weights/resnet18-f37072fd.pth",
            "torchvision_version": "0.20.1+cpu", "torchvision_module_path": "/recorded/torchvision/__init__.py",
            "environment": {"python": "3.11.9", "torch": "2.5.1+cpu", "numpy": "2.1.3",
                            "opencv": "4.10.0", "sklearn": "1.6.0", "platform": "fixture",
                            "processor": "fixture", "cpu_count": 8, "device": "cuda",
                            "cuda_runtime": "12.4", "cuda_device": "recorded GPU",
                            "torchvision": "0.20.1+cpu", "torchvision_version": "0.20.1+cpu",
                            "torchvision_module_path": "/recorded/torchvision/__init__.py"},
            "code_fingerprints": {
                name: hashlib.sha256((self.runner.PROJECT / "experiments" / name).read_bytes()).hexdigest()
                for name in ("run_resnet_study.py", "resnet_robustness_study.py", "run_c_study.py",
                             "cnn_robustness_study.py", "cnn_robustness_pilot.py")},
        }

    def write_study(self, output, matrix):
        from cnn_robustness_study import config_fingerprint, sha256_file
        common = self.official_common()
        output.mkdir()
        (output / "run_config.json").write_text(json.dumps({"common": common, "matrix": matrix}))
        (output / "split.csv").write_bytes(b"split\n")
        (output / "data_manifest.json").write_text(json.dumps({
            "data_fingerprint": common["data_fingerprint"], "source_files": [],
            "generated_fingerprints": {}, "extra_indices": [], "augmentation_counts": {},
            "preparation_seconds": 0.1}))
        for augmentation, mode, seed in matrix:
            run_id = f"{augmentation}_{mode}_seed{seed}"
            directory = output / "runs" / run_id
            directory.mkdir(parents=True)
            config = dict(common, run_id=run_id, input=mode, augmentation=augmentation, seed=seed,
                          preprocessing_seconds={"data_preparation": 0.1})
            (directory / "config.json").write_text(json.dumps(config))
            (directory / "status.json").write_text(json.dumps({"status": "complete", "run_id": run_id}))
            # Generic verification treats model/CSV/NPZ bytes as opaque, checksum-protected artifacts.
            # Actual model reload and tensor-shape checks belong to the ResNet core tests.
            for name in ("model.pt", "training_log.csv", "metrics.csv", "predictions.csv", "cost.json"):
                (directory / name).write_bytes(b"opaque-artifact\n")
            names = ["train_clean", "validation_clean", "validation_motion", "validation_noise",
                     "validation_dark", "validation_jpeg"]
            (directory / "features").mkdir()
            for name in names:
                (directory / "features" / f"{name}.npz").write_bytes(b"opaque-feature-artifact\n")
            if mode in ("dual", "dual_denoised"):
                for name in ("intervention_metrics.csv", "intervention_predictions.csv"):
                    (directory / name).write_bytes(b"opaque-intervention-artifact\n")
            artifacts = {path.relative_to(directory).as_posix(): sha256_file(path)
                         for path in directory.rglob("*") if path.is_file()}
            (directory / "completion.json").write_text(json.dumps({
                "status": "complete", "config_fingerprint": config_fingerprint(config),
                "artifacts": artifacts, "evaluations": names,
                "fusion": "fixed" if mode == "fixed" else "learned"}))
        return common

    def test_rejects_legacy_backbone_and_wrong_preprocessing_or_weights(self):
        common = self.official_common()
        self.runner.validate_resnet_config(common)
        for key, value in (("backbone", "SmallCNN"), ("feature_dim", 32),
                           ("network_input_size", 32), ("image_size", 224),
                           ("pretrained", False), ("weights_id", "DEFAULT"),
                           ("weights_sha256", "f37072fd" + "0" * 56)):
            with self.subTest(key=key):
                changed = dict(common, **{key: value})
                with self.assertRaises(ValueError):
                    self.runner.validate_resnet_config(changed)
        changed = copy.deepcopy(common)
        changed["environment"]["torchvision_version"] = "0.1.0"
        with self.assertRaises(ValueError):
            self.runner.validate_resnet_config(changed)
        for path in ("/external/resnet18.pth", "C:/external/resnet18.pth", "../resnet18.pth",
                     "assets\\weights\\resnet18.pth"):
            with self.subTest(weights_path=path):
                with self.assertRaises(ValueError):
                    self.runner.validate_resnet_config(dict(common, weights_path=path))

    def test_verify_only_checks_all_fifteen_runs_without_weights_or_cuda(self):
        matrix = [[augmentation, mode, seed] for augmentation, mode in
                  (("repeat_clean", "dual"), ("targeted_motion", "dual"), ("mixed", "dual"),
                   ("mixed", "fixed"), ("mixed", "dual_denoised")) for seed in (42, 43, 44)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            self.write_study(output, matrix)
            before = {path.relative_to(output).as_posix(): path.read_bytes()
                      for path in output.rglob("*") if path.is_file()}
            with patch("torch.cuda.is_available", side_effect=AssertionError("GPU probe in read-only verification")):
                self.runner.main(["--verify-only", "--device", "cuda", "--weights-path",
                                  str(Path(temporary) / "absent.pth"), "--output-dir", str(output)])
            self.assertEqual(before, {path.relative_to(output).as_posix(): path.read_bytes()
                                      for path in output.rglob("*") if path.is_file()})
            # The final requested run must be checked, rather than stopping at an earlier run.
            (output / "runs" / "mixed_dual_denoised_seed44" / "predictions.csv").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                self.runner.main(["--verify-only", "--output-dir", str(output)])

    def test_verify_only_rejects_partial_or_different_study_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            self.write_study(output, [["mixed", "dual", 42]])
            with self.assertRaises(ValueError):
                self.runner.main(["--verify-only", "--output-dir", str(output)])
            (output / "data_manifest.json").unlink()
            with self.assertRaises(ValueError):
                self.runner.main(["--verify-only", "--inputs", "dual", "--augmentations", "mixed",
                                  "--seeds", "42", "--output-dir", str(output)])

    def test_resume_refuses_partial_run_before_gpu_or_weight_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            self.write_study(output, [["mixed", "dual", 42]])
            (output / "runs" / "mixed_dual_seed42" / "completion.json").unlink()
            before = {path.relative_to(output).as_posix(): path.read_bytes()
                      for path in output.rglob("*") if path.is_file()}
            with patch("torch.cuda.is_available", side_effect=AssertionError("GPU probe before resume validation")):
                with self.assertRaises(ValueError):
                    self.runner.main(["--resume", "--device", "cuda", "--weights-path",
                                      str(Path(temporary) / "absent.pth"), "--inputs", "dual",
                                      "--augmentations", "mixed", "--seeds", "42", "--output-dir", str(output)])
            self.assertEqual(before, {path.relative_to(output).as_posix(): path.read_bytes()
                                      for path in output.rglob("*") if path.is_file()})

    def test_spark_runtime_reservation_initializes_without_overwriting_runtime(self):
        self.assertTrue(hasattr(self.runner, "_initialize_study"), "缺少安全的 Spark 预留目录初始化")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            output.mkdir()
            runtime = output / "spark_runtime.json"
            runtime.write_text(json.dumps({"schema_version": 1, "image_id": "sha256:" + "a" * 64,
                                           "torch": "NVIDIA-test"}))
            before = runtime.read_bytes()
            split = Path(temporary) / "source_split.csv"
            split.write_bytes(b"split\n")
            config, manifest = {"common": {}, "matrix": []}, {"data_fingerprint": "fixture"}
            self.assertTrue(self.runner._is_spark_reservation(output))
            self.runner._initialize_study(output, config, manifest, split, spark_reserved=True)
            self.assertEqual(runtime.read_bytes(), before)
            self.assertEqual(json.loads((output / "run_config.json").read_text()), config)
            self.assertEqual(json.loads((output / "data_manifest.json").read_text()), manifest)
            self.assertEqual((output / "split.csv").read_bytes(), b"split\n")
            with self.assertRaises(ValueError):
                self.runner._initialize_study(output, config, manifest, split, spark_reserved=True)

    def test_spark_reservation_rejects_invalid_record_and_unexpected_partial_files(self):
        self.assertTrue(hasattr(self.runner, "_is_spark_reservation"), "缺少 Spark 预留目录检查")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            output.mkdir()
            runtime = output / "spark_runtime.json"
            for record in ({"schema_version": 2, "image_id": "sha256:" + "a" * 64},
                           {"schema_version": 1, "image_id": "sha256:bad"}, ["invalid"]):
                runtime.write_text(json.dumps(record))
                self.assertFalse(self.runner._is_spark_reservation(output))
            runtime.write_text(json.dumps({"schema_version": 1, "image_id": "sha256:" + "a" * 64}))
            partial = output / "partial_model.pt"
            partial.write_bytes(b"do-not-overwrite")
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertFalse(self.runner._is_spark_reservation(output))
            with self.assertRaises(ValueError):
                self.runner.main(["--resume", "--output-dir", str(output)])
            self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
