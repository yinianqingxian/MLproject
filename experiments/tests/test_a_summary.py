"""
用途：核验 A 的十三条件汇总程序在真实产物结构上的完整性检查与聚合行为。
核心职责：
- 用合成的评估目录跑通读取、校验、汇总、配对、干扰族、逐类与图表生成。
- 确认缺少种子、条件不齐或调试标记的输入会被拒绝，而不是给出不完整结论。
关键副作用/安全约束：
- 只写项目内的忽略目录 `.test_tmp/`，结束后删除；不读写正式结果。
"""

import csv
import json
from pathlib import Path
import shutil
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a_conditions import LEVELS, SEVERITIES
from summarize_a_study import CONDITION_ORDER, load_eval, main as summarize_main

PROJECT = Path(__file__).resolve().parents[2]
TMP_ROOT = PROJECT / ".test_tmp"
CLASSES = (1, 2, 3, 4, 5, 7, 8)
IMAGES = 4


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _macro_f1(augmentation, condition, severity, seed):
    base = 0.80 if augmentation == "mixed" else 0.70
    penalty = {"none": 0.0, "low": 0.02, "medium": 0.05, "high": 0.10}[severity]
    return base - penalty - 0.001 * (seed - 42)


def build_eval(root, seeds=(42, 43, 44)):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for augmentation in ("repeat_clean", "mixed"):
        for seed in seeds:
            run_id = f"{augmentation}_raw_seed{seed}"
            run_dir = root / run_id
            metrics, predictions, costs = [], [], []
            for condition, severity in CONDITION_ORDER:
                score = _macro_f1(augmentation, condition, severity, seed)
                metrics.append({
                    "run_id": run_id, "seed": seed, "augmentation": augmentation, "input": "raw",
                    "split": "validation", "condition": condition, "severity": severity,
                    "macro_f1": f"{score:.8f}", "accuracy": f"{score + 0.01:.8f}",
                    "n_images": IMAGES, "parameters": 11181255,
                })
                for index in range(IMAGES):
                    label = CLASSES[index % len(CLASSES)]
                    predictions.append({
                        "run_id": run_id, "seed": seed, "split": "validation",
                        "image": f"train/{index:05d}.ppm", "track_id": "00001/00001_00000",
                        "condition": condition, "severity": severity,
                        "y_true": label, "y_pred": label if index else CLASSES[0], "gate": "",
                    })
                costs.append({
                    "condition": condition, "severity": severity, "parameter": "",
                    "images": IMAGES, "batch_size": 128,
                    "preprocessing_seconds": 0.01, "evaluation_seconds": 0.02,
                })
            _write_csv(run_dir / "metrics.csv",
                       ("run_id", "seed", "augmentation", "input", "split", "condition",
                        "severity", "macro_f1", "accuracy", "n_images", "parameters"), metrics)
            _write_csv(run_dir / "predictions.csv",
                       ("run_id", "seed", "split", "image", "track_id", "condition",
                        "severity", "y_true", "y_pred", "gate"), predictions)
            _write_csv(run_dir / "costs.csv",
                       tuple(costs[0]), costs)
            records.append({"run_id": run_id})
    (root / "completion.json").write_text(json.dumps({
        "schema_version": 1, "study": "synthetic_study", "conditions": 13,
        "images": IMAGES, "runs": records, "limited": False,
    }, indent=2), encoding="utf-8")
    return root


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = TMP_ROOT / "a_summary"
        if cls.root.exists():
            shutil.rmtree(cls.root, ignore_errors=True)
        cls.eval_dir = build_eval(cls.root / "eval")
        cls.report_dir = cls.root / "report"
        summarize_main(["--eval", str(cls.eval_dir), "--output", str(cls.report_dir)])
        with (cls.report_dir / "tables" / "dev13_summary.csv").open(encoding="utf-8") as handle:
            cls.summary = list(csv.DictReader(handle))
        with (cls.report_dir / "tables" / "dev13_paired.csv").open(encoding="utf-8") as handle:
            cls.paired = list(csv.DictReader(handle))
        with (cls.report_dir / "tables" / "dev13_families.csv").open(encoding="utf-8") as handle:
            cls.families = list(csv.DictReader(handle))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def test_tables_and_figures_cover_every_condition(self):
        self.assertEqual(len(self.summary), 2 * len(CONDITION_ORDER))
        self.assertEqual(len(self.families), 2 * (len(LEVELS) + 1))
        self.assertEqual(len(self.paired), len(CONDITION_ORDER))
        for name in ("dev13_by_run.csv", "dev13_summary.csv", "dev13_paired.csv",
                     "dev13_families.csv", "dev13_per_class.csv", "dev13_costs.csv"):
            self.assertTrue((self.report_dir / "tables" / name).is_file(), name)
        for name in ("dev13_curves.png", "dev13_conditions.png", "dev13_families.png"):
            self.assertTrue((self.report_dir / "figures" / name).is_file(), name)
        self.assertIn("十三条件", (self.report_dir / "dev13_summary.md").read_text(encoding="utf-8"))

    def test_three_seed_mean_matches_the_synthetic_scores(self):
        row = next(item for item in self.summary
                   if item["config"] == "mixed" and item["condition"] == "noise"
                   and item["severity"] == "high")
        expected = 100 * sum(_macro_f1("mixed", "noise", "high", seed) for seed in (42, 43, 44)) / 3
        self.assertAlmostEqual(float(row["macro_f1_mean"]), expected, places=4)
        self.assertEqual(int(row["seed_count"]), 3)
        self.assertGreater(float(row["macro_f1_sd"]), 0.0)

    def test_paired_difference_is_positive_and_unanimous(self):
        row = next(item for item in self.paired if item["condition"] == "clean")
        self.assertEqual(row["comparison"], "mixed - repeat_clean")
        self.assertAlmostEqual(float(row["delta_mean"]), 10.0, places=4)
        self.assertEqual(row["improved_seeds"], "3")

    def test_incomplete_seeds_or_conditions_are_rejected(self):
        broken = TMP_ROOT / "broken"
        build_eval(broken, seeds=(42, 43))
        with self.assertRaises(ValueError):
            load_eval(broken)
        (self.eval_dir / "debug_limited.json").write_text("{}", encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_eval(self.eval_dir)
        finally:
            (self.eval_dir / "debug_limited.json").unlink()


if __name__ == "__main__":
    unittest.main()
