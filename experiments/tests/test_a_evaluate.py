"""
用途：核验十三条件评估路径在真实 ResNet18 前向、指标与导出字段上的契约。
核心职责：
- 用未训练的官方 ResNet18 结构走一遍 `_evaluate`，检查输出字典、形状与类别范围。
- 检查指标行、逐图预测行与条件清单能被官方测试/报告程序直接消费。
关键副作用/安全约束：
- 只做前向推理，不训练、不写正式结果目录；不读取官方测试图像。
"""

import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a_conditions import conditions, manifest
from cnn_robustness_study import _evaluate, _metric_row, _prediction_rows, tensor_views
from run_c_study import PROJECT, prepare_data

DATA_ROOT = Path(os.environ.get("A_DATA_ROOT", PROJECT / "data/GTSRB/raw"))
SPLIT = Path(os.environ.get("A_SPLIT", PROJECT / "results/pilot_split.csv"))
CLASSES = [1, 2, 3, 4, 5, 7, 8]
IMAGES = 8
PREDICTION_FIELDS = {"run_id", "seed", "split", "image", "track_id", "condition",
                     "severity", "y_true", "y_pred", "gate"}


class EvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from resnet_robustness_study import ResNet18Fusion

        cls.data = prepare_data(DATA_ROOT, SPLIT)
        cls.rows = cls.data["validation_rows"][:IMAGES]
        # Use the public frozen-condition generator on a short prefix of the development set.
        cls.specs = [(kind, severity, value, images[:IMAGES])
                     for kind, severity, value, images in conditions(cls.data)]
        torch.set_num_threads(2)
        cls.model = ResNet18Fusion("single", pretrained=False)
        cls.model.eval()

    def test_thirteen_conditions_export_consistent_artifacts(self):
        config = {"run_id": "smoke_raw_seed42", "seed": 42, "augmentation": "mixed",
                  "input": "raw", "classes": CLASSES}
        metrics, predictions = [], []
        for kind, severity, _, images in self.specs:
            spec = {"views": tensor_views(images), "rows": self.rows, "split": "validation",
                    "condition": kind, "severity": severity}
            output, elapsed = _evaluate(self.model, "raw", spec, CLASSES, 4)
            self.assertGreater(elapsed, 0.0)
            self.assertEqual(output["features"].shape, (IMAGES, 512))
            self.assertEqual(output["y_pred"].shape, (IMAGES,))
            self.assertTrue(set(np.unique(output["y_pred"])) <= set(CLASSES))
            self.assertTrue(np.isfinite(output["features"]).all())
            metrics.append(_metric_row(config, spec, output, 11181255))
            predictions.extend(_prediction_rows(config, spec, output))
        self.assertEqual(len(metrics), 13)
        self.assertEqual(len(predictions), 13 * IMAGES)
        self.assertEqual(len({(row["condition"], row["severity"]) for row in metrics}), 13)
        for row in metrics:
            self.assertGreaterEqual(row["macro_f1"], 0.0)
            self.assertLessEqual(row["macro_f1"], 1.0)
            self.assertEqual(row["n_images"], IMAGES)
        for row in predictions:
            self.assertEqual(set(row), PREDICTION_FIELDS)
            self.assertEqual(row["gate"], "")

    def test_condition_manifest_is_fingerprinted(self):
        entries = manifest(self.specs)
        self.assertEqual(len(entries), 13)
        self.assertEqual(len({entry["pixel_sha256"] for entry in entries}), 13)
        for entry in entries:
            self.assertEqual(len(entry["pixel_sha256"]), 64)
            self.assertEqual(entry["n_images"], IMAGES)


if __name__ == "__main__":
    unittest.main()
