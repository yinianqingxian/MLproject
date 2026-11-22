"""
用途：核验 ResNet18 汇总的产物身份、特征维度和独立统计。
核心职责：
- 拒绝历史骨干、错误预训练来源和非 512 维特征。
- 使用手工可复核样例检查七类指标，并验证失败时不生成报告。
关键副作用/安全约束：
- 只在临时目录构造测试产物，不训练模型或修改正式结果。
"""

import csv
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ResNetSummaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("summarize_resnet_study"),
                             "独立 ResNet18 汇总入口尚未实现")
        self.summary = importlib.import_module("summarize_resnet_study")
        self.config = {
            "run_id": "mixed_dual_seed42", "augmentation": "mixed", "input": "dual", "seed": 42,
            "classes": [1, 2, 3, 4, 5, 7, 8], "epochs": 10, "batch_size": 128,
            "learning_rate": .003, "image_size": 32, "optimizer": "AdamW", "weight_decay": .01,
            "betas": [.9, .999], "eps": 1e-8, "training_entries": 4410,
            "clean_training_images": 2940, "validation_images": 840,
            "train_tracks": 98, "validation_tracks": 28,
            "clahe": {"channel": "LAB_L", "clip_limit": 2.0, "tile_grid": [4, 4]},
            "denoising": {"method": "bilateral", "d": 5, "sigmaColor": 25, "sigmaSpace": 2},
            "data_fingerprint": {"sha256": "fixture"},
            "backbone": "torchvision.models.resnet18", "pretrained": True,
            "feature_dim": 512, "network_input_size": 224,
            "weights_id": "ResNet18_Weights.IMAGENET1K_V1",
            "weights_path": "assets/weights/resnet18-f37072fd.pth",
            "weights_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            "torchvision_version": "0.20.1", "torchvision_module_path": "/training/torchvision/__init__.py",
            "preprocessing_seconds": {}, "schema_version": 1, "threads": 6, "device": "cpu",
            "evaluation": "final_epoch_development_validation", "split_sha256": "b" * 64,
            "code_fingerprints": {name: "c" * 64 for name in (
                "run_resnet_study.py", "resnet_robustness_study.py", "run_c_study.py",
                "cnn_robustness_study.py", "cnn_robustness_pilot.py")},
            "environment": {"python": "3.12", "torch": "2.5.1", "numpy": "2.1.3", "opencv": "4.10",
                            "sklearn": "1.6", "platform": "test", "processor": "test", "cpu_count": 6,
                            "device": "cpu", "cuda_runtime": None, "cuda_device": None,
                            "torchvision": "0.20.1", "torchvision_version": "0.20.1",
                            "torchvision_module_path": "/training/torchvision/__init__.py"},
        }
        self.config["data_fingerprint"] = "a" * 64

    def test_official_configuration_is_accepted_without_importing_torchvision(self):
        before = sys.modules.get("torchvision")
        self.summary._validate_config(self.config, Path(self.config["run_id"]))
        self.assertIs(sys.modules.get("torchvision"), before)

    def test_incomplete_official_runtime_provenance_is_rejected(self):
        invalid = dict(self.config)
        del invalid["torchvision_module_path"]
        with self.assertRaises(self.summary.StudyValidationError):
            self.summary._validate_config(invalid, Path(invalid["run_id"]))

    def test_historical_backbone_and_32_features_are_rejected(self):
        for field, value in (("backbone", "SmallCNN"), ("feature_dim", 32),
                             ("network_input_size", 32), ("pretrained", False)):
            with self.subTest(field=field):
                invalid = {**self.config, field: value}
                with self.assertRaises(self.summary.StudyValidationError):
                    self.summary._validate_config(invalid, Path(invalid["run_id"]))

    def test_official_pretraining_identity_cannot_be_substituted(self):
        for field, value in (("weights_id", "ResNet18_Weights.DEFAULT"),
                             ("weights_sha256", "a" * 64),
                             ("weights_path", "../resnet18-f37072fd.pth"),
                             ("weights_path", "C:/download/resnet18-f37072fd.pth"),
                             ("torchvision_version", "")):
            with self.subTest(field=field, value=value):
                invalid = {**self.config, field: value}
                with self.assertRaises(self.summary.StudyValidationError):
                    self.summary._validate_config(invalid, Path(invalid["run_id"]))

    def test_seven_class_metrics_and_sample_sd_are_independently_recomputed(self):
        f1, accuracy = self.summary.classification_scores([1, 2, 3, 4, 5, 7, 8], [1, 1, 3, 4, 5, 7, 8])
        self.assertAlmostEqual(f1, 17 / 21)
        self.assertAlmostEqual(accuracy, 6 / 7)
        self.assertEqual(self.summary.mean_sd([1, 2, 3]), (2., 1.))

    def test_csv_scores_are_checked_against_predictions(self):
        predictions, metrics = [], []
        for condition in ("clean", "motion", "noise", "dark", "jpeg"):
            predictions.extend({"condition": condition, "severity": "fixture", "y_true": truth, "y_pred": predicted}
                               for _ in range(120)
                               for truth, predicted in zip((1, 2, 3, 4, 5, 7, 8), (1, 1, 3, 4, 5, 7, 8)))
            metrics.append({"run_id": self.config["run_id"], "seed": 42, "augmentation": "mixed", "input": "dual",
                            "split": "validation", "condition": condition, "severity": "fixture", "n_images": 840,
                            "parameters": 11_000_000, "macro_f1": 17 / 21, "accuracy": 6 / 7})
        self.summary._validate_metrics(metrics, predictions, self.config)
        metrics[0]["macro_f1"] = .8
        with self.assertRaisesRegex(self.summary.StudyValidationError, "recomputation"):
            self.summary._validate_metrics(metrics, predictions, self.config)

    @staticmethod
    def _split_metadata():
        return {split: sorted(
            (f"{split}_{label}_{track:02d}_{frame:02d}.png", f"{split}_{label}_{track:02d}", label)
            for label in (1, 2, 3, 4, 5, 7, 8)
            for track in range(tracks_per_class) for frame in range(30))
            for split, tracks_per_class in (("train", 14), ("validation", 4))}

    def _feature_fixture(self, directory):
        reference = self._split_metadata()
        predictions = []
        feature_dir = directory / "features"
        feature_dir.mkdir()
        for name in ("train_clean", "validation_clean", "validation_motion", "validation_noise",
                     "validation_dark", "validation_jpeg"):
            split, condition = name.split("_", 1)
            metadata = reference[split]
            np.savez(feature_dir / f"{name}.npz", image=[row[0] for row in metadata],
                     track_id=[row[1] for row in metadata], y_true=[row[2] for row in metadata],
                     y_pred=[row[2] for row in metadata], gate=np.full(len(metadata), .4),
                     features=np.zeros((len(metadata), 512), dtype=np.float32))
            if split == "validation":
                predictions.extend({"condition": condition, "image": image, "track_id": track,
                                    "y_true": label, "y_pred": label, "gate": .4}
                                   for image, track, label in metadata)
        return predictions, reference

    def test_512_features_keep_all_split_and_prediction_alignment_checks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            directory = Path(temporary)
            predictions, reference = self._feature_fixture(directory)
            gates = self.summary._validate_features(directory, self.config, predictions, reference)
            self.assertEqual(len(gates), 48)
            self.assertEqual(gates[0]["n_images"], 2940)
            path = directory / "features" / "validation_clean.npz"
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            data["features"] = np.zeros((840, 32), dtype=np.float32)
            np.savez(path, **data)
            with self.assertRaisesRegex(self.summary.StudyValidationError, "512"):
                self.summary._validate_features(directory, self.config, predictions, reference)

    def test_empty_study_does_not_create_report_or_aggregate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            study = Path(temporary) / "study"
            report = Path(temporary) / "report"
            (study / "runs").mkdir(parents=True)
            with self.assertRaises(self.summary.StudyValidationError):
                self.summary.main(["--study-dir", str(study), "--report-dir", str(report)])
            self.assertFalse(report.exists())
            self.assertFalse((study / "metrics.csv").exists())

    def test_historical_manifest_is_rejected_before_loading_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            study = Path(temporary) / "study"
            report = Path(temporary) / "report"
            for index in range(15):
                (study / "runs" / f"run{index}").mkdir(parents=True)
            common = {**self.config, "backbone": "SmallCNN"}
            matrix = [[aug, mode, seed] for aug, mode in self.summary.CONFIGURATIONS for seed in (42, 43, 44)]
            (study / "run_config.json").write_text(json.dumps({"common": common, "matrix": matrix}), encoding="utf-8")
            with self.assertRaisesRegex(self.summary.StudyValidationError, "backbone|ResNet18|resnet18"):
                self.summary.main(["--study-dir", str(study), "--report-dir", str(report)])
            self.assertFalse(report.exists())

    def test_extra_per_run_common_field_cannot_escape_frozen_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            study = Path(temporary) / "study"
            report = Path(temporary) / "report"
            study.mkdir()
            common, _ = self._study_metadata_fixture(study)
            matrix = [[aug, mode, seed] for aug, mode in self.summary.CONFIGURATIONS for seed in (42, 43, 44)]
            for augmentation, mode, seed in matrix:
                run_id = f"{augmentation}_{mode}_seed{seed}"
                directory = study / "runs" / run_id
                directory.mkdir(parents=True)
                config = {**common, "run_id": run_id, "seed": seed, "augmentation": augmentation,
                          "input": mode, "preprocessing_seconds": {}}
                if run_id == "mixed_dual_seed42":
                    config["unexpected_common_field"] = True
                (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (study / "run_config.json").write_text(json.dumps({"common": common, "matrix": matrix}), encoding="utf-8")
            with self.assertRaisesRegex(self.summary.StudyValidationError, "common configuration mismatch"):
                self.summary.main(["--study-dir", str(study), "--report-dir", str(report)])
            self.assertFalse(report.exists())

    def _study_metadata_fixture(self, study):
        common = {key: value for key, value in self.config.items() if key not in
                  ("run_id", "seed", "augmentation", "input", "preprocessing_seconds")}
        split_rows = [(split, image, label, track) for split, metadata in self._split_metadata().items()
                      for image, track, label in metadata]
        with (study / "split.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("split", "image", "class_id", "track_id"))
            writer.writerows(split_rows)
        common["split_sha256"] = hashlib.sha256((study / "split.csv").read_bytes()).hexdigest()
        source_files = [{"image": row[1], "sha256": "d" * 64} for row in split_rows]
        generated = {key: "e" * 64 for key in (
            "training/mixed", "training/repeat_clean", "training/targeted_motion",
            "validation/clean", "validation/dark", "validation/jpeg", "validation/motion", "validation/noise")}
        # Reproduce the documented trainer serialization, independently of the summary.
        fingerprint = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode())
        fingerprint.update(json.dumps(split_rows, separators=(",", ":")).encode())
        for key, value in generated.items():
            fingerprint.update(f"{key}:{value}".encode())
        common["data_fingerprint"] = fingerprint.hexdigest()
        manifest = {"source_files": source_files, "generated_fingerprints": generated,
                    "extra_indices": list(range(1470)),
                    "augmentation_counts": {"repeat_clean": {"repeat": 1470}, "targeted_motion": {"motion": 1470},
                                            "mixed": {"motion": 490, "noise": 490, "dark": 490}},
                    "data_fingerprint": common["data_fingerprint"], "preparation_seconds": 1.}
        (study / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return common, manifest

    def _metadata_only_study(self, study):
        study.mkdir()
        common, manifest = self._study_metadata_fixture(study)
        matrix = [[aug, mode, seed] for aug, mode in self.summary.CONFIGURATIONS for seed in (42, 43, 44)]
        for augmentation, mode, seed in matrix:
            run_id = f"{augmentation}_{mode}_seed{seed}"
            directory = study / "runs" / run_id
            directory.mkdir(parents=True)
            config = {**common, "run_id": run_id, "seed": seed, "augmentation": augmentation,
                      "input": mode, "preprocessing_seconds": {}}
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (study / "run_config.json").write_text(json.dumps({"common": common, "matrix": matrix}), encoding="utf-8")
        return common, manifest

    def test_missing_or_changed_study_data_manifest_stops_before_report(self):
        for mutation in ("missing", "missing_key", "different_data_fingerprint", "changed_generated_fingerprint"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
                study, report = Path(temporary) / "study", Path(temporary) / "report"
                _, manifest = self._metadata_only_study(study)
                path = study / "data_manifest.json"
                if mutation == "missing":
                    path.unlink()
                else:
                    if mutation == "missing_key":
                        del manifest["source_files"]
                    elif mutation == "different_data_fingerprint":
                        manifest["data_fingerprint"] = "f" * 64
                    else:
                        manifest["generated_fingerprints"]["training/mixed"] = "f" * 64
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(self.summary.StudyValidationError, "data.manifest|data fingerprint"):
                    self.summary.main(["--study-dir", str(study), "--report-dir", str(report)])
                self.assertFalse(report.exists())
                self.assertFalse((study / "metrics.csv").exists())

    def test_missing_or_byte_changed_study_split_stops_before_report(self):
        for mutation in ("missing", "byte_change"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
                study, report = Path(temporary) / "study", Path(temporary) / "report"
                self._metadata_only_study(study)
                path = study / "split.csv"
                if mutation == "missing":
                    path.unlink()
                else:
                    # CSV parsing ignores this newline, but the frozen byte identity must change.
                    path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(self.summary.StudyValidationError, "split.csv|split.*fingerprint"):
                    self.summary.main(["--study-dir", str(study), "--report-dir", str(report)])
                self.assertFalse(report.exists())

    def test_study_data_fingerprint_recomputes_without_current_code_or_raw_images(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            study = Path(temporary)
            common, _ = self._study_metadata_fixture(study)
            self.assertTrue(hasattr(self.summary, "_validate_study_metadata"), "研究级数据核验尚未实现")
            reference = self.summary._validate_study_metadata(study, common)
            self.assertEqual(len(reference["train"]), 2940)
            self.assertEqual(len(reference["validation"]), 840)

    def _report_text(self, temporary, *, gate, endpoint_prediction, noise_gain):
        study = Path(temporary) / "study"
        runs = []
        for augmentation, mode in self.summary.CONFIGURATIONS:
            for seed in (42, 43, 44):
                run_id = f"{augmentation}_{mode}_seed{seed}"
                config = {**self.config, "run_id": run_id, "augmentation": augmentation, "input": mode, "seed": seed}
                metrics, predictions, interventions, intervention_predictions, gates = [], [], [], [], []
                for condition in ("clean", "motion", "noise", "dark", "jpeg"):
                    score = .6 + (noise_gain if mode == "dual_denoised" and condition == "noise" else 0)
                    metric = {"run_id": run_id, "seed": seed, "augmentation": augmentation, "input": mode,
                              "condition": condition, "severity": "fixture", "macro_f1": score, "accuracy": score,
                              "parameters": 11_000_000}
                    metrics.append(metric)
                    predictions.append({"condition": condition, "image": "fixture.png", "y_pred": 1,
                                        "gate": .5 if mode == "fixed" else gate})
                    gates.append({"split": "validation", "class_id": "all", "augmentation": augmentation,
                                  "input": mode, "condition": condition, "gate_mean": gate, "q05": gate, "q95": gate})
                    if mode != "fixed":
                        for override in (0., .5, 1.):
                            interventions.append({**metric, "gate_override": override})
                            intervention_predictions.append({"condition": condition, "image": "fixture.png",
                                                             "gate_override": override,
                                                             "y_pred": endpoint_prediction if override == 1. else 1})
                training = [{"epoch": epoch, "train_loss": 1., "validation_macro_f1": .6,
                             "validation_accuracy": .6, "epoch_seconds": 1.} for epoch in range(1, 11)]
                cost = {"parameters": 11_000_000, "training_seconds": 10., "export_seconds": 1.,
                        "training_entries": 4410, "optimizer_steps": 350, "threads": 6,
                        "inference_seconds": {name: 1. for name in ("train_clean", "validation_clean", "validation_motion",
                                                                    "validation_noise", "validation_dark", "validation_jpeg")}}
                runs.append(self.summary.Run(study / "runs" / run_id, config, metrics, predictions, interventions,
                                             intervention_predictions, training, cost, gates))
        metrics = [row for run in runs for row in run.metrics]
        summary = self.summary.summarize_metrics(metrics)
        paired, paired_summary = self.summary.paired_differences(metrics)
        _, interventions = self.summary.summarize_interventions(runs)
        _, convergence, costs, _ = self.summary.convergence_and_cost(runs)
        return self.summary.build_report(study, Path(temporary) / "report", runs, summary, paired, paired_summary,
                                         interventions, convergence, costs, self.summary.summarize_costs(costs))

    def test_report_conclusions_follow_current_scores_and_unsaturated_gates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            report = self._report_text(temporary, gate=.4, endpoint_prediction=1, noise_gain=.05)
        self.assertIn("预设描述性噪声收益检查满足", report)
        self.assertIn("门控最小值为 0.400000", report)
        self.assertIn("官方预训练 ResNet18", report)
        self.assertIn("512 维表示", report)
        self.assertNotIn("最小 g 为 1.000000", report)

    def test_saturated_gates_with_changed_endpoint_predictions_do_not_claim_collapse(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            report = self._report_text(temporary, gate=1., endpoint_prediction=2, noise_gain=0.)
        self.assertIn("预设描述性噪声收益检查未全部满足", report)
        self.assertIn("g=1 干预共改变 45 条预测", report)
        self.assertNotIn("全部门控 g≥0.99 且 g=1 干预不改变预测", report)


if __name__ == "__main__":
    unittest.main()
