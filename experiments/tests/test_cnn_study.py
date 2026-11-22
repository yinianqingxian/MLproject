"""
用途：核验融合实验的科学控制、特征对齐与结果保存行为。
核心职责：
- 检查共享编码器、门控端点和梯度传递。
- 使用小型真实训练验证权重重载、元数据对齐和恢复校验。
关键副作用/安全约束：
- 仅在临时目录保存测试运行，不覆盖正式实验。
"""

import copy
import csv
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FusionStudyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("cnn_robustness_study"),
                             "公共融合训练模块尚未实现")
        self.study = importlib.import_module("cnn_robustness_study")
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_gate_endpoints_match_corresponding_encoder_and_classifier(self):
        model = self.study.SmallCNN("learned").eval()
        raw, enhanced = torch.rand(4, 3, 32, 32), torch.rand(4, 3, 32, 32)
        with torch.inference_mode():
            raw_expected = model.classifier(model.encoder(raw))
            enhanced_expected = model.classifier(model.encoder(enhanced))
            torch.testing.assert_close(model(raw, enhanced, gate_override=0.0), raw_expected)
            torch.testing.assert_close(model(raw, enhanced, gate_override=1.0), enhanced_expected)
            details = model.forward_details(raw, enhanced)
        torch.testing.assert_close(details["gate"], torch.full((4, 1), 0.5))
        self.assertEqual(tuple(details["features"].shape), (4, 32))

    def test_shared_encoder_and_gate_receive_gradients(self):
        model = self.study.SmallCNN("learned")
        raw, enhanced = torch.rand(4, 3, 32, 32), torch.rand(4, 3, 32, 32)
        loss = torch.nn.functional.cross_entropy(model(raw, enhanced), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertGreater(float(model.gate.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()), 0)
        self.assertEqual(sum(isinstance(layer, torch.nn.Conv2d) for layer in model.modules()), 3)
        fixed = self.study.SmallCNN("fixed").eval()
        with torch.inference_mode():
            expected = fixed.classifier((fixed.encoder(raw) + fixed.encoder(enhanced)) / 2)
            torch.testing.assert_close(fixed(raw, enhanced), expected)

    def test_view_adapter_preserves_original_and_converts_bgr_to_rgb(self):
        source = np.zeros((32, 32, 3), dtype=np.uint8)
        source[:, :, 0], source[:, :, 1], source[:, :, 2] = 10, 70, 200
        original = source.copy()
        views = self.study.tensor_views([source])
        np.testing.assert_array_equal(source, original)
        torch.testing.assert_close(views["raw"][0, :, 0, 0], torch.tensor([200, 70, 10]) / 255)
        for value in views.values():
            self.assertEqual(tuple(value.shape), (1, 3, 32, 32))
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertGreaterEqual(float(value.min()), 0)
            self.assertLessEqual(float(value.max()), 1)

    def test_training_exports_sorted_features_and_reloadable_predictions(self):
        images = torch.rand(14, 3, 32, 32)
        views = {"raw": images, "clahe": 1 - images, "denoised_clahe": 1 - images}
        rows = [{"split": "validation", "image": f"{13-i:05d}.ppm", "class_id": [1,2,3,4,5,7,8][i % 7],
                 "track_id": f"track-{i}"} for i in range(14)]
        train_rows = [dict(row, split="train", image="train/" + row["image"]) for row in rows]
        evaluations = {
            "train_clean": {"views": views, "rows": train_rows, "split": "train", "condition": "clean"},
            "validation_clean": {"views": views, "rows": rows, "split": "validation", "condition": "clean"},
        }
        config = {"schema_version": 1, "run_id": "mixed_dual_seed42", "seed": 42, "input": "dual",
                  "augmentation": "mixed", "epochs": 1, "batch_size": 7, "learning_rate": 0.003,
                  "classes": [1,2,3,4,5,7,8], "data_fingerprint": "test-data"}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / config["run_id"]
            self.study.fit_run(run_dir, config, views, torch.arange(14) % 7, evaluations)
            self.assertTrue(self.study.verify_completed_run(run_dir, config))
            checkpoint = torch.load(run_dir / "model.pt", weights_only=True)
            model = self.study.SmallCNN("learned")
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            with np.load(run_dir / "features" / "validation_clean.npz", allow_pickle=False) as exported:
                order = sorted(range(14), key=lambda i: rows[i]["image"])
                self.assertEqual(exported["image"].tolist(), sorted(row["image"] for row in rows))
                self.assertEqual(exported["track_id"].tolist(), [rows[i]["track_id"] for i in order])
                self.assertEqual(exported["y_true"].tolist(), [rows[i]["class_id"] for i in order])
                with torch.inference_mode():
                    actual = model.forward_details(images[order], (1 - images)[order])
                np.testing.assert_allclose(exported["features"], actual["features"].numpy(), atol=1e-6)
                predicted = np.array(config["classes"])[actual["logits"].argmax(1).numpy()]
                np.testing.assert_array_equal(exported["y_pred"], predicted)
            with (run_dir / "predictions.csv").open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), 14)
            with self.assertRaises(FileExistsError):
                self.study.fit_run(run_dir, config, views, torch.arange(14) % 7, evaluations)
            changed = copy.deepcopy(config)
            changed["epochs"] = 2
            with self.assertRaises(ValueError):
                self.study.verify_completed_run(run_dir, changed)
            with (run_dir / "predictions.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaises(ValueError):
                self.study.verify_completed_run(run_dir, config)


if __name__ == "__main__":
    unittest.main()
