"""
用途：核验 A 的冻结十三条件、官方测试子集清单与干扰生成的可复现性。
核心职责：
- 检查开发集条件数量、顺序、像素预算，以及五个条件与 prepare_data 完全一致。
- 检查官方测试子集由标注文件确定，规模与类别分布固定。
- 检查低/中/高三档噪声共享同一标准正态噪声场。
关键副作用/安全约束：
- 只读数据与元数据；临时文件写在项目内的忽略目录，不触碰正式结果。
"""

import csv
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a_conditions
from a_conditions import LEVELS, NOISE_SEED, TEST_NOISE_SEED, conditions, pixel_hash, test_conditions
from run_c_study import PROJECT, prepare_data

DATA_ROOT = Path(os.environ.get("A_DATA_ROOT", PROJECT / "data/GTSRB/raw"))
SPLIT = Path(os.environ.get("A_SPLIT", PROJECT / "results/pilot_split.csv"))


class DevelopmentConditionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepare_data(DATA_ROOT, SPLIT)

    def test_thirteen_frozen_conditions_repeat_exactly(self):
        first = list(conditions(self.data))
        second = list(conditions(self.data))
        self.assertEqual(len(first), 13)
        self.assertEqual(len({(kind, severity) for kind, severity, _, _ in first}), 13)
        self.assertEqual(first[0][0:2], ("clean", "none"))
        self.assertEqual([kind for kind, _, _, _ in first],
                         ["clean"] + [kind for kind in LEVELS for _ in a_conditions.SEVERITIES])
        for (kind, severity, value, images), repeat in zip(first, second):
            self.assertEqual(len(images), 840)
            self.assertEqual(pixel_hash(images), pixel_hash(repeat[3]))
            self.assertEqual(np.asarray(images).shape, (840, 32, 32, 3))
            if severity in ("none", "medium"):
                # The legacy clean and medium pixels must be reused bit for bit.
                legacy = (self.data["validation"]["clean"] if kind == "clean"
                          else self.data["validation"][kind])
                np.testing.assert_array_equal(images, legacy)

    def test_medium_parameters_match_the_frozen_table(self):
        specs = {(kind, severity): value for kind, severity, value, _ in conditions(self.data)}
        for kind, values in LEVELS.items():
            for severity, value in zip(a_conditions.SEVERITIES, values):
                self.assertEqual(specs[(kind, severity)], value)


class TestConditionGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mid-grey avoids clipping so the shared normal noise field stays visible.
        cls.clean = [np.full((32, 32, 3), 128, dtype=np.uint8) for _ in range(6)]
        cls.rows = [{"image": f"GTSRB/Final_Test/Images/{index:05d}.ppm", "class_id": 1,
                     "track_id": "", "split": "test"} for index in range(6)]

    def test_all_thirteen_conditions_are_deterministic(self):
        first = list(test_conditions(self.clean, self.rows))
        second = list(test_conditions(self.clean, self.rows))
        self.assertEqual(len(first), 13)
        for (kind, severity, value, images), repeat in zip(first, second):
            self.assertEqual(len(images), 6)
            self.assertEqual(pixel_hash(images), pixel_hash(repeat[3]))
            self.assertEqual(np.asarray(images).shape, (6, 32, 32, 3))
        self.assertEqual(first[0][0:2], ("clean", "none"))
        np.testing.assert_array_equal(first[0][3], self.clean)

    def test_noise_strengths_share_one_normal_field(self):
        specs = {(kind, severity): images for kind, severity, _, images in
                 test_conditions(self.clean, self.rows)}
        clean = np.asarray(self.clean, dtype=np.float64)
        low = np.asarray(specs[("noise", "low")], dtype=np.float64) - clean
        for severity, sigma in zip(a_conditions.SEVERITIES, LEVELS["noise"]):
            deviate = np.asarray(specs[("noise", severity)], dtype=np.float64) - clean
            # Same normal field rescaled by sigma; compare only pixels that were not clipped.
            unclipped = (low > -128) & (low < 127) & (deviate > -128) & (deviate < 127)
            self.assertGreater(unclipped.mean(), 0.99)
            np.testing.assert_allclose(deviate[unclipped],
                                       (low * sigma / LEVELS["noise"][0])[unclipped], atol=3.5)
            self.assertGreater(np.abs(deviate).mean(), sigma / 4)
        self.assertNotEqual(NOISE_SEED, TEST_NOISE_SEED)

    def test_motion_direction_alternates_on_frozen_order(self):
        impulse = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
        for image in impulse:
            image[16, 16] = 255
        specs = {(kind, severity): images for kind, severity, _, images in
                 test_conditions(impulse, self.rows[:4])}
        high = np.asarray(specs[("motion", "high")])
        # Kernel size 7: even rows blur horizontally, odd rows blur vertically.
        self.assertEqual(int(np.count_nonzero(high[0, 16, :, 0])), 7)
        self.assertEqual(int(np.count_nonzero(high[0, :, 16, 0])), 1)
        self.assertEqual(int(np.count_nonzero(high[1, :, 16, 0])), 7)
        self.assertEqual(int(np.count_nonzero(high[1, 16, :, 0])), 1)

    def test_jpeg_and_dark_preserve_the_output_contract(self):
        specs = {(kind, severity): images for kind, severity, _, images in
                 test_conditions(self.clean, self.rows)}
        for severity, quality in zip(a_conditions.SEVERITIES, LEVELS["jpeg"]):
            images = np.asarray(specs[("jpeg", severity)])
            self.assertEqual(images.shape, (6, 32, 32, 3))
            self.assertEqual(images.dtype, np.uint8)
            self.assertTrue(np.abs(images.astype(int) - 128).mean() < 12)
        for severity, gain in zip(a_conditions.SEVERITIES, LEVELS["dark"]):
            images = np.asarray(specs[("dark", severity)]).astype(int)
            np.testing.assert_allclose(images, int(128 * gain), atol=1)


class OfficialTestSubsetTests(unittest.TestCase):
    def test_subset_matches_the_official_labels(self):
        from evaluate_official_test import EXPECTED_IMAGES, _test_rows

        rows = _test_rows(DATA_ROOT / "GT-final_test.csv")
        self.assertEqual(len(rows), sum(EXPECTED_IMAGES.values()))
        counts = {}
        for row in rows:
            counts[row["class_id"]] = counts.get(row["class_id"], 0) + 1
        self.assertEqual(counts, EXPECTED_IMAGES)
        self.assertEqual([row["image"] for row in rows], sorted(row["image"] for row in rows))
        self.assertTrue(all(row["image"].startswith("GTSRB/Final_Test/Images/") for row in rows))
        self.assertTrue(all(row["split"] == "test" for row in rows))

    def test_training_split_never_contains_test_images(self):
        with SPLIT.open(encoding="utf-8-sig", newline="") as handle:
            training = {row["image"] for row in csv.DictReader(handle)}
        self.assertTrue(all(image.startswith("GTSRB/Final_Training/Images/") for image in training))


if __name__ == "__main__":
    unittest.main()
