# C 部分交接说明

截至 2026-11-17，历史 SmallCNN C 范围已完成。用户现已确认保留双视图融合、仅将骨干替换为官方 ResNet18；当前默认 Spark 入口已转向新方案，新的正式训练仍待执行。新协议见 [03-resnet-c-implementation.md](03-resnet-c-implementation.md)，下文旧分数、32 维特征和 15 个模型属于历史 SmallCNN。

用户确认接收机器为 DGX Spark（ARM64/GB10），现在优先按 [DGX_SPARK.md](DGX_SPARK.md) 运行专用 NVIDIA 容器；本机仍只实际完成 Windows CPU 训练，不将 Spark 配置验证写成目标机器已训练。

## 已完成的材料

原 `experiments/cnn_robustness_pilot.py` 和预实验结果未修改。新增 `cnn_robustness_study.py` 提供公共模型、六种输入、训练、干预及导出；`run_c_study.py` 提供冻结数据与实验矩阵；`summarize_c_study.py` 校验逐图证据、生成统计表和报告。

正式结果在 `results/study_v1/runs/`：15 个模型、75 条指标、63,000 条常规预测、90 份特征，以及 12 个门控模型的 180 条干预指标和 151,200 条干预预测。`results/study_v1/checkpoint_verification.json` 记录重载核验通过：107,100 张图像的特征/预测/门控与全部干预预测一致。`results/c_smoke/` 是六种模式各一轮的工程核验，不用于性能结论。

## 历史 SmallCNN 已交付范围

1. 提供全组公共训练入口。原图、CLAHE 单分支和共享编码器融合都使用同样的小 CNN、训练参数和结果导出格式。
2. 实现门控融合与独立训练的固定平均融合。门控融合为 `(1-g) * z_raw + g * z_enhanced`，两个视图共用编码器，特征为 32 维。
3. 保存最终权重、逐轮损失、逐图预测、真实类别、图像名、轨迹、32 维特征和门控值。特征按图像名排序，便于 D 的分析。
4. 跑门控双分支的无干扰重复采样、针对性模糊、混合增强，每种三个种子，共 9 次训练。
5. 在混合增强下再跑固定平均融合、原图 + 去噪 CLAHE 门控融合，各三个种子，共 6 次训练。
6. 对同一门控模型固定 g 为 0、0.5、1 做干预。与“从头训练的固定平均模型”分别报告。
7. 汇总门控分布、重复实验波动、计算代价，完成 C 的模型与融合消融报告章节。

## 固定协议

- 七类为 1、2、3、4、5、7、8，已有 `pilot_split.csv` 不更换。
- 干净训练 2,940 张，开发验证 840 张，分别来自 98 和 28 条轨迹。
- 各组增加 1,470 个条目，共 4,410 个条目。无干扰组重复对应干净图，混合组各 490 张模糊、噪声与变暗。
- 32×32 输入，batch size 128，AdamW，学习率 0.003，固定 10 轮，种子 42、43、44，评价最后一轮。
- CLAHE 仅处理 LAB 的 L 通道，clip limit 2.0，网格 4×4。
- 去噪视图采用双边滤波 `d=5, sigmaColor=25, sigmaSpace=2`，随后再做 CLAHE。该参数为待检验初始方案。
- 模型编码器的三个通道层为 3→12→24→32，卷积、GroupNorm、ReLU、MaxPool，每张图输出 32 维特征。详见现有预实验代码。

## 与 A、B、D 的接口

A/B 的计划模块当前不存在。先复用预实验的分组与干扰函数，保持原有随机生成顺序；C 程序中的临时适配函数以后可替换为 A/B 的实现。不要提前覆盖他们负责的文件。

公共训练结果应按 `results/study_v1/runs/<run_id>/` 分开保存。运行编号包含增强策略、输入模式和种子。拒绝静默覆盖，恢复已完成运行需核验配置与产物指纹。D 需要逐图元数据和特征，而不仅是 Macro-F1 汇总值。

## 核验和重新运行

原六项 C 测试（四项模型/导出及两项真实数据协议）已通过；补充 Spark 验收后，全套 19 项测试通过。双击 `RUN_C_TESTS.cmd`，或在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\experiments\tests -v
.\.venv\Scripts\python.exe .\experiments\run_c_study.py --verify-only
.\.venv\Scripts\python.exe .\experiments\summarize_c_study.py
```

这些测试覆盖融合端点、共享编码器梯度、颜色空间、权重重载、特征对齐、篡改检测、预实验随机样本一致性与划分修改拒绝。受限沙箱若无法使用系统临时目录，可使用 `RUN_C_TESTS.cmd`，它会将 TEMP/TMP 指向项目 `.test_tmp`。

重新训练请指定新目录，例如 `--output-dir results/study_v2`；对已完成目录使用 `--resume` 前会核对配置、源码、数据、环境及全部产物指纹。半成品不能从中断轮次续训，保留诊断信息并换新目录重跑。汇总会更新派生报告，使用自定义研究目录时同时指定新的 `--report-dir`。

## 结果与交接边界

混合增强门控模型的 clean/motion/noise Macro-F1 均值为 87.98%/76.70%/77.86%。相对针对性模糊增强，噪声增加 8.43 pp、模糊降低 1.82 pp。普通门控几乎完全选择 CLAHE，不能称为已验证的自适应融合；去噪融合在当前协议下退步，保留该负结果。

A/B 仍需用公共入口生成正式单分支对照，D 可使用每运行 `features/train_clean.npz` 与五个 `validation_*.npz` 做分析。PCA/KMeans 仅在干净训练特征拟合。C 的 g=0/1 干预不能代替从头训练的单分支。官方测试、多强度和全 43 类均不属于本次已完成结论。

## 给另一台电脑上的协作者

可以把下面这段直接交给协助开发的 AI。

> 历史 SmallCNN C 已完成。当前按用户确认改用官方 ResNet18 共享骨干，保留原融合与消融。目标机器为 DGX Spark，请阅读 DGX_SPARK.md 和 03-resnet-c-implementation.md，运行 bash SPARK.sh setup、smoke、train、verify、report。保持原数据和划分，结果写入 results/study_resnet18_spark_v1，特征为 512 维。新 ResNet 正式训练尚未执行，不能沿用旧分数；官方测试仍不用于开发调参。
