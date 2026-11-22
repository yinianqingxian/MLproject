"""
用途：核验官方 ResNet18 融合模型与独立实验产物的真实性。
核心职责：
- 检查官方预训练骨干、逐批预处理、门控端点和共享梯度。
- 以短程真实训练核验 512 维特征、模型重载及配置指纹。
关键副作用/安全约束：
- 仅在临时目录写入训练产物，不下载权重或覆盖正式实验。
"""

import copy
import csv
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ResNetStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torchvision
        except (ImportError, RuntimeError, OSError) as error:
            raise unittest.SkipTest(f"Matched torch/torchvision is unavailable: {error}")
        cls.vision = torchvision

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("resnet_robustness_study"),
                             "官方 ResNet18 融合模块尚未实现")
        self.study = importlib.import_module("resnet_robustness_study")
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_official_backbone_uses_full_local_imagenet_state(self):
        model = self.study.ResNet18Fusion("learned")
        self.assertIsInstance(model.encoder, self.vision.models.ResNet)
        self.assertIsInstance(model.encoder.fc, torch.nn.Identity)
        self.assertIsInstance(model.encoder.bn1, torch.nn.BatchNorm2d)
        self.assertEqual(model.encoder.conv1.kernel_size, (7, 7))
        self.assertEqual(model.encoder.conv1.stride, (2, 2))
        self.assertEqual(model.feature_dim, 512)
        self.assertEqual((model.classifier.in_features, model.classifier.out_features), (512, 7))
        self.assertEqual(model.gate.in_features, 1024)
        source = torch.load(self.study.DEFAULT_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        actual = model.encoder.state_dict()
        source.pop("fc.weight")
        source.pop("fc.bias")
        added = set(actual) - set(source)
        self.assertTrue(all(name.endswith(".num_batches_tracked") for name in added))
        self.assertTrue(set(source).issubset(actual))
        for name, value in actual.items():
            torch.testing.assert_close(value, source.get(name, torch.tensor(0)), rtol=0, atol=0)

    def test_gate_endpoints_use_official_preprocessing_once_on_concatenated_views(self):
        model = self.study.ResNet18Fusion("learned", pretrained=False).eval()
        raw, enhanced = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
        transform = self.vision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        with torch.inference_mode():
            raw_features, enhanced_features = model.encoder(transform(torch.cat((raw, enhanced)))).chunk(2)
            torch.testing.assert_close(model(raw, enhanced, gate_override=0.0),
                                       model.classifier(raw_features))
            torch.testing.assert_close(model(raw, enhanced, gate_override=1.0),
                                       model.classifier(enhanced_features))
            seen = []
            hook = model.encoder.register_forward_pre_hook(lambda _model, args: seen.append(args[0]))
            try:
                details = model.forward_details(raw, enhanced)
            finally:
                hook.remove()
        self.assertEqual(len(seen), 1)
        self.assertEqual(tuple(seen[0].shape), (4, 3, 224, 224))
        torch.testing.assert_close(seen[0], transform(torch.cat((raw, enhanced))), rtol=0, atol=0)
        torch.testing.assert_close(details["gate"], torch.full((2, 1), 0.5))
        torch.testing.assert_close(details["features"], (raw_features + enhanced_features) / 2)
        self.assertEqual(tuple(details["features"].shape), (2, 512))

    def test_single_and_fixed_branches_preserve_existing_fusion_contract(self):
        raw, enhanced = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
        transform = self.vision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        single = self.study.ResNet18Fusion(pretrained=False).eval()
        with torch.inference_mode():
            details = single.forward_details(raw)
            torch.testing.assert_close(details["features"], single.encoder(transform(raw)))
        self.assertTrue(bool(torch.isnan(details["gate"]).all()))
        with self.assertRaises(ValueError):
            single(raw, enhanced)
        fixed = self.study.ResNet18Fusion("fixed", pretrained=False).eval()
        self.assertIsNone(fixed.gate)
        with torch.inference_mode():
            left, right = fixed.encoder(transform(torch.cat((raw, enhanced)))).chunk(2)
            torch.testing.assert_close(fixed(raw, enhanced), fixed.classifier((left + right) / 2))
        for override in (-0.1, 1.1, float("nan")):
            with self.subTest(override=override), self.assertRaises(ValueError):
                fixed(raw, enhanced, gate_override=override)

    def test_learned_gate_and_shared_official_encoder_receive_gradients(self):
        model = self.study.ResNet18Fusion("learned", pretrained=False)
        raw, enhanced = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
        loss = torch.nn.functional.cross_entropy(model(raw, enhanced), torch.tensor([0, 1]))
        loss.backward()
        self.assertGreater(float(model.gate.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.encoder.conv1.weight.grad.abs().sum()), 0)
        self.assertEqual(sum(isinstance(layer, self.vision.models.ResNet)
                             for layer in model.modules()), 1)
        self.assertTrue(all(parameter.requires_grad for parameter in model.encoder.parameters()))

    def _smoke_inputs(self):
        images = torch.rand(4, 3, 32, 32)
        views = {"raw": images, "clahe": 1 - images, "denoised_clahe": 1 - images}
        rows = [{"split": "validation", "image": f"{3-i:05d}.ppm", "class_id": [1, 2, 3, 4][i],
                 "track_id": f"track-{i}"} for i in range(4)]
        train_rows = [dict(row, split="train", image="train/" + row["image"]) for row in rows]
        evaluations = {
            "train_clean": {"views": views, "rows": train_rows, "split": "train", "condition": "clean"},
            "validation_clean": {"views": views, "rows": rows, "split": "validation", "condition": "clean"},
        }
        config = {"schema_version": 1, "run_id": "smoke_dual_seed42", "seed": 42, "input": "dual",
                  "augmentation": "mixed", "epochs": 1, "batch_size": 2, "learning_rate": 0.003,
                  "classes": [1, 2, 3, 4, 5, 7, 8], "data_fingerprint": "smoke-data",
                  "image_size": 32, "threads": 2, "device": "cpu",
                  **self.study.weights_metadata(pretrained=False)}
        return config, views, rows, evaluations

    def test_training_exports_sorted_512_features_and_reloadable_cpu_checkpoint(self):
        config, views, rows, evaluations = self._smoke_inputs()
        self.assertFalse(config["pretrained"])
        self.assertIsNone(config["weights_id"])
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / config["run_id"]
            self.study.fit_run(run_dir, config, views, torch.arange(4), evaluations)
            self.assertTrue(self.study.verify_completed_run(run_dir, config))
            checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["config"], config)
            self.assertTrue(all(value.device.type == "cpu" for value in checkpoint["model_state"].values()))
            model = self.study.ResNet18Fusion("learned", pretrained=False).eval()
            model.load_state_dict(checkpoint["model_state"])
            order = sorted(range(4), key=lambda index: rows[index]["image"])
            with torch.inference_mode():
                actual = [model.forward_details(views["raw"][indices], views["clahe"][indices])
                          for indices in (order[:2], order[2:])]
            with np.load(run_dir / "features" / "validation_clean.npz", allow_pickle=False) as exported:
                self.assertEqual(exported["features"].shape, (4, 512))
                self.assertEqual(exported["raw_features"].shape, (4, 512))
                self.assertEqual(exported["enhanced_features"].shape, (4, 512))
                self.assertEqual(exported["image"].tolist(), sorted(row["image"] for row in rows))
                self.assertEqual(exported["track_id"].tolist(), [rows[index]["track_id"] for index in order])
                self.assertEqual(exported["y_true"].tolist(), [4, 3, 2, 1])
                np.testing.assert_allclose(exported["features"],
                                           torch.cat([part["features"] for part in actual]).numpy(),
                                           rtol=1e-4, atol=2e-5)
                predicted = np.array(config["classes"])[torch.cat([part["logits"] for part in actual]).argmax(1)]
                np.testing.assert_array_equal(exported["y_pred"], predicted)
            with (run_dir / "training_log.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual([row["epoch"] for row in csv.DictReader(handle)], ["1"])
            with (run_dir / "intervention_metrics.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual([row["gate_override"] for row in csv.DictReader(handle)], ["0.0", "0.5", "1.0"])
            cost = json.loads((run_dir / "cost.json").read_text(encoding="utf-8"))
            self.assertEqual((cost["optimizer_steps"], cost["training_entries"]), (2, 4))
            with self.assertRaises(FileExistsError):
                self.study.fit_run(run_dir, config, views, torch.arange(4), evaluations)
            changed = copy.deepcopy(config)
            changed["network_input_size"] = 32
            with self.assertRaises(ValueError):
                self.study.verify_completed_run(run_dir, changed)
            with (run_dir / "predictions.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaises(ValueError):
                self.study.verify_completed_run(run_dir, config)

    def test_false_weight_provenance_is_rejected_before_creating_run(self):
        config, views, _rows, evaluations = self._smoke_inputs()
        official = self.study.weights_metadata()
        self.assertEqual(len(official["weights_sha256"]), 64)
        self.assertEqual(official["weights_path"], "assets/weights/resnet18-f37072fd.pth")
        for change in ({"backbone": "SmallCNN"}, {"feature_dim": 32}, {"network_input_size": 32},
                       {"weights_id": "ResNet18_Weights.IMAGENET1K_V1"},
                       {**official, "weights_sha256": "0" * 64}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "invalid"
                with self.assertRaises(ValueError):
                    self.study.fit_run(run_dir, dict(config, **change), views, torch.arange(4), evaluations)
                self.assertFalse(run_dir.exists())


if __name__ == "__main__":
    unittest.main()
