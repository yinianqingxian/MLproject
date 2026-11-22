"""
用途：验证 B 到 C 交接审计能拒绝损坏资料并正确汇总三个种子的对照。
核心职责：
- 构造独立的协议样本，覆盖矩阵、分数、预算和来源指纹校验。
- 检查配对统计及已有输出目录保护，避免把结构验收当成训练验收。
关键副作用/安全约束：
- 仅在临时目录生成样本和审计输出，不执行交接脚本或训练。
"""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2]
TOOL = PROJECT / "tools/audit_b_handoff.py"
WEIGHTS = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
CODE = ("run_resnet_study.py", "resnet_robustness_study.py", "run_c_study.py",
        "cnn_robustness_study.py", "cnn_robustness_pilot.py")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class HandoffAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.exists(), "The independent B handoff audit tool is not implemented")
        spec = importlib.util.spec_from_file_location("audit_b_handoff", TOOL)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.project = self.root / "handoff", self.root / "project"
        self.source.mkdir()
        fingerprints = {}
        for name in CODE:
            path = self.project / "experiments" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
            fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        split = []
        for class_id in (1, 2, 3, 4, 5, 7, 8):
            for track in range(18):
                for frame in range(30):
                    split.append({"split": "train" if track < 14 else "validation",
                                  "image": f"images/{class_id}/{track}_{frame}.ppm",
                                  "class_id": str(class_id), "track_id": f"{class_id}/{track}"})
        write_csv(self.source / "清单/pilot_split.csv", split)
        write_csv(self.project / "results/pilot_split.csv", split)
        write_csv(self.source / "清单/images.csv", [r for r in split if r["split"] == "validation"])
        self.conditions = [("clean", "none", None)] + [
            (kind, severity, parameter) for kind, values in
            (("motion", (3, 5, 7)), ("noise", (10, 20, 30)),
             ("dark", (0.7, 0.45, 0.25)), ("jpeg", (40, 20, 10)))
            for severity, parameter in zip(("low", "medium", "high"), values)]
        condition_rows = [{"condition": k, "severity": s, "parameter": p,
                           "n_images": 840, "pixel_sha256": "a" * 64} for k, s, p in self.conditions]
        split_sha = hashlib.sha256((self.source / "清单/pilot_split.csv").read_bytes()).hexdigest()
        write_json(self.source / "清单/conditions.json", {
            "levels": {"motion": [3, 5, 7], "noise": [10, 20, 30],
                       "dark": [0.7, 0.45, 0.25], "jpeg": [40, 20, 10]},
            "noise_seed": 20261117, "data_fingerprint": "b" * 64,
            "split_sha256": split_sha, "conditions": condition_rows})
        write_json(self.source / "成员B样例/输入核对_Mac.json", {
            "opencv": "5.1.0", "conditions": [
                {"condition": k, "severity": s, "count": 840, "local_input": "a" * 64,
                 "a_reference_input": ("a" if i < 4 else "c") * 64, "matches_a": i < 4}
                for i, (k, s, _) in enumerate(self.conditions)]})
        self.metrics = []
        for augmentation, mode in (("repeat_clean", "clahe"), ("targeted_motion", "clahe"),
                                   ("mixed", "clahe"), ("mixed", "denoised_clahe")):
            for seed in (42, 43, 44):
                run = f"{augmentation}_{mode}_seed{seed}"
                config = {"run_id": run, "augmentation": augmentation, "input": mode,
                          "seed": seed, "backbone": "torchvision.models.resnet18",
                          "feature_dim": 512, "network_input_size": 224, "image_size": 32,
                          "classes": [1, 2, 3, 4, 5, 7, 8], "epochs": 10, "batch_size": 128,
                          "learning_rate": 0.003, "optimizer": "AdamW", "weight_decay": 0.01,
                          "training_entries": 4410, "clean_training_images": 2940,
                          "validation_images": 840, "train_tracks": 98, "validation_tracks": 28,
                          "clahe": {"clip_limit": 2.0, "tile_grid": [4, 4], "channel": "LAB_L"},
                          "denoising": {"method": "bilateral", "d": 5, "sigmaColor": 25, "sigmaSpace": 2},
                          "pretrained": True, "weights_id": "ResNet18_Weights.IMAGENET1K_V1",
                          "weights_sha256": WEIGHTS, "code_fingerprints": fingerprints,
                          "data_fingerprint": "b" * 64, "split_sha256": split_sha}
                write_json(self.source / f"配置/{run}/config.json", config)
                write_csv(self.source / f"计时/{run}/costs.csv", [
                    {"condition": k, "severity": s, "preprocessing_seconds": 0.1,
                     "evaluation_seconds": 2.0, "images": 840} for k, s, _ in self.conditions])
                for k, s, _ in self.conditions:
                    self.metrics.append({"run_id": run, "seed": seed, "augmentation": augmentation,
                                         "input": mode, "split": "validation", "condition": k,
                                         "severity": s, "macro_f1": 0.7 + (seed - 42) * 0.1 -
                                         (0.1 if mode == "denoised_clahe" else 0),
                                         "accuracy": 0.8, "n_images": 840, "parameters": 11180103})
        self.metric_path = self.source / "B_13条件指标汇总.csv"
        write_csv(self.metric_path, self.metrics)

    def test_valid_structure_reports_limits_and_paired_sample_statistics(self):
        result = self.tool.audit_handoff(self.source, self.project)
        self.assertEqual(result["counts"]["metric_rows"], 156)
        self.assertFalse(result["safe_for_cross_member_paired_comparison"])
        self.assertFalse(result["training_fully_verified"])
        self.assertEqual(result["input_reference"]["matching_conditions"], 4)
        tables = result["tables"]
        self.assertEqual([len(tables[k]) for k in ("b_metric_summary", "b_mixed_denoising_pairs",
                                                 "b_mixed_denoising_summary")], [52, 39, 13])
        self.assertAlmostEqual(tables["b_metric_summary"][0]["macro_f1_sd"], 0.1)
        self.assertAlmostEqual(tables["b_mixed_denoising_summary"][0]["macro_f1_delta_mean"], -0.1)

    def test_rejects_missing_metric(self):
        write_csv(self.metric_path, self.metrics[:-1])
        with self.assertRaisesRegex(ValueError, "metric"):
            self.tool.audit_handoff(self.source, self.project)

    def test_rejects_duplicate_metric_even_with_correct_total(self):
        write_csv(self.metric_path, self.metrics[:-1] + [self.metrics[0]])
        with self.assertRaisesRegex(ValueError, "metric"):
            self.tool.audit_handoff(self.source, self.project)

    def test_rejects_nonfinite_or_out_of_range_score(self):
        for value in ("nan", "inf", "-0.1", "1.1"):
            with self.subTest(value=value):
                rows = [dict(r) for r in self.metrics]
                rows[0]["macro_f1"] = value
                write_csv(self.metric_path, rows)
                with self.assertRaisesRegex(ValueError, "macro_f1"):
                    self.tool.audit_handoff(self.source, self.project)

    def test_rejects_duplicate_score_header_that_hides_nonfinite_value(self):
        fields = list(self.metrics[0])
        with self.metric_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields + ["macro_f1"])
            for index, row in enumerate(self.metrics):
                values = ["nan" if index == 0 and key == "macro_f1" else row[key]
                          for key in fields]
                writer.writerow(values + [row["macro_f1"]])
        with self.assertRaisesRegex(ValueError, "Duplicate CSV"):
            self.tool.audit_handoff(self.source, self.project)

    def test_rejects_extra_csv_cell_instead_of_discarding_none_key(self):
        fields = list(self.metrics[0])
        with self.metric_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for index, row in enumerate(self.metrics):
                values = [row[key] for key in fields]
                writer.writerow(values + (["silently ignored"] if index == 0 else []))
        with self.assertRaisesRegex(ValueError, "CSV row width"):
            self.tool.audit_handoff(self.source, self.project)

    def test_rejects_conflicting_training_budget(self):
        path = next((self.source / "配置").glob("*/config.json"))
        config = json.loads(path.read_text())
        config["training_entries"] = 4409
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, "training_entries"):
            self.tool.audit_handoff(self.source, self.project)

    def test_rejects_cost_missing_condition(self):
        path = next((self.source / "计时").glob("*/costs.csv"))
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        write_csv(path, rows[:-1])
        with self.assertRaisesRegex(ValueError, "cost"):
            self.tool.audit_handoff(self.source, self.project)

    def test_rejects_input_audit_that_disagrees_with_condition_pixels(self):
        path = self.source / "成员B样例/输入核对_Mac.json"
        audit = json.loads(path.read_text())
        audit["conditions"][0]["local_input"] = "d" * 64
        write_json(path, audit)
        with self.assertRaisesRegex(ValueError, "Mac input audit"):
            self.tool.audit_handoff(self.source, self.project)

    def test_rejects_image_track_mismatch(self):
        path = self.source / "清单/images.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["track_id"] = "wrong_track"
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "images/track"):
            self.tool.audit_handoff(self.source, self.project)

    def test_source_manifest_detects_tampering(self):
        files = [{"path": p.relative_to(self.source).as_posix(), "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in self.source.rglob("*") if p.is_file()]
        write_json(self.source / "SOURCE_MANIFEST.json", {"files": files})
        self.metric_path.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.tool.audit_handoff(self.source, self.project)

    def test_cli_writes_four_outputs_and_refuses_existing_directory(self):
        output = self.root / "output"
        self.assertEqual(self.tool.main(["--source", str(self.source), "--project", str(self.project),
                                        "--output", str(output)]), 0)
        self.assertEqual(len(list(output.iterdir())), 4)
        before = (output / "audit.json").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "exist"):
            self.tool.main(["--source", str(self.source), "--project", str(self.project),
                            "--output", str(output)])
        self.assertEqual((output / "audit.json").read_bytes(), before)

    def test_cli_refuses_to_write_inside_readonly_handoff(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.tool.main(["--source", str(self.source), "--project", str(self.project),
                            "--output", str(self.source / "new_output")])


if __name__ == "__main__":
    unittest.main()
