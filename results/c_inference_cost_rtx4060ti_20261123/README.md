# C 部分同机推理成本测量

在 Windows RTX 4060 Ti 上，缓存视图后的单分支评估平均约 0.69–0.71 毫秒/图，门控双分支平均约 1.41–1.43 毫秒/图。五个开发条件下，双分支约为单分支的 2.00–2.05 倍。这是 batch 128 的吞吐折算。详见 [配对汇总](summary.csv)、[45 条原始计时](timings.csv)和[环境及指纹](benchmark_manifest.json)。

## 测量方法

每个条件使用固定验证集中的相同 840 张图。原图单路、CLAHE 单路与原图加 CLAHE 门控双路共用 `mixed_dual_seed42` 的 ResNet18 编码器和分类头权重。每种输入预热 128 张，再计时三次。每轮轮换执行顺序，记录同轮的双路减单路差值。使用 FP32、6 个 CPU 线程，采用 CUDA 同步计时。程序还核对了单路视图的 logits 与同一融合模型对应固定端点的 logits。

计时从已生成的 CPU 视图缓存开始，包含批次取图、设备传输、官方输入变换、网络前向和结果回传。三种视图共用缓存的构建耗时单独记在环境清单。该测量未分摊每种视图独立生成的时间，不能用作逐张请求的在线延迟。

## 与 A 的对照核验

当前单路权重来自 C，结论限于同权重下单路和双路的执行成本。取得 A 的正式 `mixed/raw/seed42` 运行目录后，使用同一脚本的 `--raw-run` 参数加入 A 的权重，重新生成一份同机对照。该入口要求训练预算、种子、划分和数据指纹匹配。B 的 Mac CPU 结果保留为其原环境记录。

在项目根目录选择一个未使用的输出目录执行。使用配对的 CUDA PyTorch 与 Torchvision 环境，准备 GTSRB 和已验收的 C 正式运行目录。数据获取见 [项目 README](../../README.md)，运行目录见 [C 交接说明](../../handoffs/C_RESNET18_HANDOFF.md)。

```bash
python tools/benchmark_resnet_inference.py --fusion-run results/study_resnet18_rtx4060ti_v1/runs/mixed_dual_seed42 --output results/c_inference_cost_rtx4060ti_recheck
```

此测量只执行评估，没有新增训练或官方测试成绩。
