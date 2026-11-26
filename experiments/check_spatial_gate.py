"""
用途：在正式训练前验证新增的空间门控（dual_spatial）实现。
核心职责：
- 检查特征与输出的形状、初始权重图是否处处 0.5、门控是否可获得梯度。
- 检查 eval 模式下 W=0 / W=1 是否分别还原单路特征，W=0.5 时两种计算顺序是否等价。
- 记录三组的参数量，确认两种门控参数量相同。
关键副作用/安全约束：
- 只前向/反向一次小批次，不写任何训练目录。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnn_robustness_study import INPUT_MODES  # noqa: E402
from resnet_robustness_study import ResNet18Fusion, weights_metadata  # noqa: E402
from run_c_study import PROJECT  # noqa: E402

WEIGHTS = PROJECT / "assets/weights/resnet18-f37072fd.pth"
CLASSES = 7


def parameters(model):
    total = sum(p.numel() for p in model.parameters())
    gate = sum(p.numel() for p in model.gate.parameters()) if model.gate is not None else 0
    return total, gate


def main():
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| 官方权重:", weights_metadata(True, WEIGHTS)["weights_sha256"][:12])

    single = ResNet18Fusion("single", pretrained=True, weights_path=WEIGHTS).to(device).eval()
    learned = ResNet18Fusion("learned", pretrained=True, weights_path=WEIGHTS).to(device).eval()
    spatial = ResNet18Fusion("spatial", pretrained=True, weights_path=WEIGHTS).to(device).eval()

    print()
    print("=== 参数量 ===")
    for name, model in (("S single", single), ("G learned", learned), ("P spatial", spatial)):
        total, gate = parameters(model)
        print(f"  {name:<12} 总参数 {total:>12,}   门控 {gate:>6,}")
    p_total, p_gate = parameters(spatial)
    g_total, g_gate = parameters(learned)
    assert p_gate == g_gate == 1025, (p_gate, g_gate)
    print("  两种门控参数量相同（1025）")

    print()
    print("=== 形状与初始化 ===")
    images = torch.rand(8, 3, 32, 32, device=device)
    enhanced = torch.rand(8, 3, 32, 32, device=device)
    with torch.inference_mode():
        out = spatial.forward_details(images, enhanced)
        raw_map = spatial._layer4_map(images)
    print("  layer4 特征图:", tuple(raw_map.shape))
    print("  权重图      :", tuple(out["gate_map"].shape))
    print("  池化后特征  :", tuple(out["features"].shape))
    print("  logits      :", tuple(out["logits"].shape))
    assert tuple(raw_map.shape) == (8, 512, 7, 7)
    assert tuple(out["gate_map"].shape) == (8, 1, 7, 7)
    assert tuple(out["features"].shape) == (8, 512)
    assert tuple(out["logits"].shape) == (8, CLASSES)
    initial = out["gate_map"]
    print(f"  初始权重图: 最小值 {float(initial.min()):.6f}，最大值 {float(initial.max()):.6f}")
    assert torch.allclose(initial, torch.full_like(initial, 0.5), atol=1e-7), "初始权重不是 0.5"

    print()
    print("=== eval 模式：W 固定为 0 / 1 / 0.5 ===")
    # 注意：cuDNN 的累加顺序随 batch 大小变化，同一模型 batch 8 与 batch 16 的特征可差 1e-3。
    # 因此参照必须用同一 batch 组成（两路拼批后取对应半），否则测到的是浮点差异而不是实现错误。
    with torch.inference_mode():
        joined_pooled = single.encoder(single.preprocess(torch.cat((images, enhanced))))
        raw_ref, enh_ref = joined_pooled[:8], joined_pooled[8:]
        alone = single.encoder(single.preprocess(images))
        print(f"  参照：同模型 batch8 与 batch16 的差（浮点口径基线）: "
              f"{float((alone - raw_ref).abs().max()):.3e}")

        zero = spatial.forward_details(images, enhanced, gate_override=0.0)
        one = spatial.forward_details(images, enhanced, gate_override=1.0)
        half = spatial.forward_details(images, enhanced, gate_override=0.5)
        d0 = float((zero["features"] - raw_ref).abs().max())
        d1 = float((one["features"] - enh_ref).abs().max())
        print(f"  W=0 时与单路原图特征的最大差        : {d0:.3e}")
        print(f"  W=1 时与单路增强图特征的最大差      : {d1:.3e}")
        assert d0 < 1e-5 and d1 < 1e-5, (d0, d1)

        fused_then_pooled = half["features"]
        pool_then_fuse = 0.5 * zero["features"] + 0.5 * one["features"]
        d2 = float((fused_then_pooled - pool_then_fuse).abs().max())
        logits_diff = float((spatial.classifier(fused_then_pooled)
                             - spatial.classifier(pool_then_fuse)).abs().max())
        print(f"  W=0.5 先融合再池化 vs 分别池化再平均: 特征差 {d2:.3e}，logit 差 {logits_diff:.3e}")
        assert d2 < 1e-5 and logits_diff < 1e-5

    print()
    print("=== 干预模式：每张图 49 个权重取平均 ===")
    with torch.inference_mode():
        normal = spatial.forward_details(images, enhanced)
        averaged = spatial.forward_details(images, enhanced, spatial_average=True)
        spread = normal["gate_map"].std(dim=(2, 3)).mean()
        print(f"  正常权重图的空间标准差（均值）: {float(spread):.6e}")
        print(f"  平均后与正常值的门控差（均值）: {float((normal['gate'] - averaged['gate']).abs().mean()):.3e}")
        print(f"  平均后权重图是否处处相同      : "
              f"{bool(torch.allclose(averaged['gate_map'], averaged['gate'].reshape(-1,1,1,1).expand_as(averaged['gate_map'])))}")

    print()
    print("=== 门控可获梯度（放在最后：train 模式会更新 BatchNorm 统计量） ===")
    trainable = ResNet18Fusion("spatial", pretrained=True, weights_path=WEIGHTS).to(device).train()
    loss = torch.nn.functional.cross_entropy(
        trainable.forward_details(images, enhanced)["logits"],
        torch.randint(0, CLASSES, (8,), device=device))
    loss.backward()
    grad = trainable.gate.weight.grad
    print(f"  门控梯度范数: {float(grad.norm()):.6e}  |  偏置梯度: {float(trainable.gate.bias.grad.abs()):.6e}")
    assert grad is not None and float(grad.norm()) > 0, "门控没有梯度"

    print()
    print("=== 输入模式登记 ===")
    for mode in ("denoised_clahe", "dual_denoised", "dual_spatial"):
        print(f"  {mode:<16}{INPUT_MODES[mode]}")
    print()
    print("全部检查通过")


if __name__ == "__main__":
    main()
