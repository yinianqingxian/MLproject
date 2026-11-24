"""
用途：核验正式融合实验沿用既有数据划分与干扰样本。
核心职责：
- 检查重复采样预算、轨迹隔离与旧预实验的随机样本一致性。
- 拒绝被修改或包含数据泄漏的划分。
"""

import copy
import csv
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cnn_robustness_pilot as pilot


class DataProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("run_c_study") is None:
            raise AssertionError("正式运行的数据适配模块尚未实现")
        cls.runner = importlib.import_module("run_c_study")
        project = Path(__file__).resolve().parents[2]
        cls.data_root = Path(os.environ.get("A_DATA_ROOT", project / "data/GTSRB/raw"))
        cls.split = Path(os.environ.get("A_SPLIT", project / "results/pilot_split.csv"))

    def test_augmentation_matches_pilot_and_clean_repeat_uses_same_sources(self):
        data = self.runner.prepare_data(self.data_root, self.split)
        rng = np.random.default_rng(42)
        rows = pilot.select_split(self.data_root, rng)
        indices = copy.deepcopy(rng).choice(2940, 1470, replace=False)
        old_training, old_y, old_validation, old_valid_y = pilot.prepare_images(cv2, self.data_root, rows, rng)
        self.assertEqual([len(data["training"][name]) for name in
                          ("repeat_clean", "targeted_motion", "mixed")], [4410, 4410, 4410])
        self.assertEqual(len(data["train_rows"]), 2940)
        self.assertEqual(len(data["validation_rows"]), 840)
        self.assertEqual(data["train_targets"], old_y)
        self.assertEqual(data["validation_targets"], old_valid_y)
        for position, source in enumerate(indices):
            np.testing.assert_array_equal(data["training"]["repeat_clean"][2940 + position],
                                          data["training"]["repeat_clean"][source])
        for name in old_training:
            np.testing.assert_array_equal(np.stack(data["training"][name]), np.stack(old_training[name]))
        for name in old_validation:
            np.testing.assert_array_equal(np.stack(data["validation"][name]), np.stack(old_validation[name]))
        self.assertEqual(data["augmentation_counts"]["mixed"], {"motion": 490, "noise": 490, "dark": 490})
        self.assertEqual(len(data["data_fingerprint"]), 64)

    def test_modified_split_is_rejected_before_training(self):
        with self.split.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["track_id"] = rows[-1]["track_id"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                self.runner.prepare_data(self.data_root, path)


if __name__ == "__main__":
    unittest.main()
