# 官方预训练 ResNet18：C 部分结果汇总与复现

状态：完整十五次运行已通过本汇总程序的配置、完整性、特征对齐和指标重算检查。本目录只覆盖固定开发验证；不代表官方测试通过，设备与软件环境影响复现数值。此 README 仅在完整矩阵核验后生成；尚未完成正式训练时，本程序不会生成结果报告。

面向接手 C 产物的小组成员。模型使用 torchvision 官方 `resnet18`、`ResNet18_Weights.IMAGENET1K_V1`，共享官方骨干，保留既有双视图融合；不将融合写成新发明。导出特征为 512 维。先在 32×32 源视图上生成干扰与 CLAHE，再应用官方 `Resize(256) → CenterCrop(224) → ImageNet Normalize`。骨干、预训练与网络输入尺寸同时改变，历史 SmallCNN 结果单独保留，不能当作隔离架构因素的比较。

## 本次训练环境与复现

本次 15 次训练在 Windows 上使用 NVIDIA GeForce RTX 4060 Ti 和 CUDA 完成，共用同一冻结开发划分。仓库保存报告、图表、指标、逐图预测及核验清单。完整权重、512 维特征和每运行的原始验证记录保存在训练机器上。

按[项目准备说明](../../README.md)取得 GTSRB 与官方初始化权重，并使用支持 CUDA 的 Python 环境，版本见下方记录。在 PowerShell 的项目根目录运行以下命令。复现路径应是新目录。

```powershell
$env:NVIDIA_TF32_OVERRIDE = '0'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python experiments/run_resnet_study.py --device cuda --epochs 10 --threads 6 --output-dir results/study_resnet18_rtx4060ti_v1_reproduction
python experiments/run_resnet_study.py --device cuda --epochs 10 --output-dir results/study_resnet18_rtx4060ti_v1_reproduction --verify-only
python experiments/summarize_resnet_study.py --study-dir results/study_resnet18_rtx4060ti_v1_reproduction --report-dir report/c_part_resnet18_rtx4060ti_reproduction
```

命令按顺序执行。设备和软件版本会影响数值及计时。已完整运行的本地结果可以直接核验和汇总。

```json
{
  "cpu_count": 16,
  "cuda_device": "NVIDIA GeForce RTX 4060 Ti",
  "cuda_runtime": "12.4",
  "device": "cuda",
  "numpy": "1.26.4",
  "opencv": "4.10.0",
  "platform": "Windows-11-10.0.26200-SP0",
  "processor": "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel",
  "python": "3.12.7",
  "sklearn": "1.5.1",
  "torch": "2.5.1",
  "torchvision": "0.20.1",
  "torchvision_version": "0.20.1"
}
```

官方初始化文件为 `assets/weights/resnet18-f37072fd.pth`，完整 SHA-256 为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

## 输入、输出与验收

输入矩阵为 dual × repeat_clean/targeted_motion/mixed，加 mixed × fixed/dual_denoised，各种子 42、43、44。必须恰好十五次运行，配置的全部公共字段与研究 `run_config.json` 完全一致，包含官方骨干、预训练、权重校验和、512 维与 224 输入元数据。

| 每运行产物 | 验收 |
| --- | --- |
| config.json、model.pt、完成指纹 | 官方 ResNet18 身份与冻结配置、源文件完整性 |
| metrics.csv、predictions.csv | 五条件、4,200 条预测，同一批 840 张、28 条轨迹；固定七类重算 Macro-F1/准确率 |
| training_log.csv | 十轮，末轮 clean 指标与导出一致 |
| features/*.npz | 六份：2,940 张 train_clean 及五份 840 张验证集；512 维，图像/轨迹/真值/预测/gate 对齐 |
| intervention_metrics.csv、intervention_predictions.csv | learned 模型 15 条指标、12,600 条预测，g=0、0.5、1；fixed 无此产物 |
| cost.json | 实际参数、训练/导出/前向耗时、4,410 条目、350 步及线程 |

前向成本包含批次选择、设备复制、模型内部官方 Resize/CenterCrop/Normalize、ResNet 计算与 CPU 输出收集；只排除源 PPM 读取及 32×32 干扰/CLAHE 视图缓存构建。配置中的 `preprocessing_seconds` 是可跨运行复用的 CPU 视图构建记录，不将它按每个模型重复累加。

研究目录必须保留自身的 `split.csv` 与 `data_manifest.json`；不以外部划分文件替代缺失快照。汇总核对 split 的完整字节 SHA-256，并从清单中记录的源文件哈希、原始 split 顺序和三种训练/五种验证像素哈希重算稳定数据指纹，不读取原始图像或重新生成干扰。成功应显示 15 次运行、75 条常规指标与 63,000 条常规预测，另外核验 180 条干预指标、151,200 条干预预测和 90 份 NPZ。三种子样本标准差使用 `ddof=1`，配对差值逐种子计算；帧级门控统计不作 IID 推断。

程序仅在全部来源通过检查后重建派生文件：

- 研究根部 `metrics.csv`、`predictions.csv` 及两份 intervention 总表。
- `tables/`：指标均值和 SD、逐种子配对差值、门控分布、同权重干预、训练轨迹和实际成本。
- `figures/`：scores_with_sd, paired_changes, gate_distribution, training_curves, intervention_effects，各有 PNG 与 SVG。
- [C_REPORT.md](C_REPORT.md)：模型、实测结果和证据边界；[summary_manifest.json](summary_manifest.json)：源文件 SHA-256、官方权重、核验计数及输出清单。

报告生成会重建派生文件，每次原始运行只读。来源缺失、历史骨干、预训练身份不符、非 512 维或预测不对齐时，以非零状态停止，不生成部分矩阵报告。绘图或文件写入失败后，保留所有输入并修复运行环境，再用相同汇总命令重建；以最后写出的 `summary_manifest.json` 作为完整生成标志。

## 失败处理与边界

数量/公共配置不符时，C 应检查原始运行与冻结清单。指纹、权重校验和或指标重算失败时停止使用对应产物，在独立输出目录重新训练，不手改来源文件。门控或元数据不一致时检查导出排序、原始类别编号及轨迹标识，不能重新排列标签掩盖错误。

本研究未提供正式单分支对照、多强度评估、独立测试或总体置信区间；同权重端点干预不等于独立训练分支。A/B 补齐正式对照，D 如需 bootstrap 应按整条轨迹配对重采样。协议依据为 `03-resnet-c-implementation.md` 及其保留的 C 矩阵；历史 SmallCNN 成果不合并进本表。
