# C 部分 ResNet18 正式产物交接

交接材料供 D 做特征聚类和统计核验，也供 A 在统一评估时读取最终模型。训练设备为 Windows 上的 RTX 4060 Ti。五种配置各训练三个种子（42、43、44），每组 10 轮。固定开发验证集共 840 张图、28 条验证轨迹。 [正式报告](../report/c_part_resnet18_rtx4060ti/C_REPORT.md) · [发布提交](https://github.com/yinianqingxian/MLproject/commit/563df2ea5b126c44745223cd55cb708d564fa2f5)

## 解压后从哪里开始

向 C 领取 `DS4023_C_ResNet18_RTX4060Ti_Results_Handoff_20261123.zip`（约 1.18 GiB）和旁附 `.sha256`。权重与特征通过组内渠道交接。先用 PowerShell 的 `Get-FileHash -Algorithm SHA256` 或 Linux 的 `sha256sum -c` 核对，再解压到新目录。

| 路径 | 用途 |
| --- | --- |
| `results/study_resnet18_rtx4060ti_v1/split.csv` | 冻结划分，含图名、类别和轨迹 ID |
| `results/study_resnet18_rtx4060ti_v1/run_config.json` 与 `data_manifest.json` | 冻结训练矩阵、预算与数据指纹 |
| `results/study_resnet18_rtx4060ti_v1/runs/<run_id>/model.pt` | 15 份末轮权重，使用 C 已发布的模型定义及该组 `config.json` 加载 |
| `results/study_resnet18_rtx4060ti_v1/runs/<run_id>/features/` | 每组 6 份 NPZ，共 90 份 |
| `results/study_resnet18_rtx4060ti_v1/` 下的四份汇总 CSV | 75 条常规指标、63,000 条验证预测、180 条干预指标、151,200 条干预预测 |
| `report/c_part_resnet18_rtx4060ti/` | GitHub 已发布的报告、图、表与汇总核验清单 |
| `HANDOFF_MANIFEST.json` | 包内各文件的大小与 SHA256 |

运行目录中的 `completion.json` 记录各原始文件的 SHA256。权重、配置和特征均原样保留，其中配置内的 Torchvision 本机安装路径是训练时的环境记录，无需在另一台机器上复现该路径。查看模型结构与训练入口可使用正式报告所在的提交。

## D 如何读取特征

混合增强的三种配置为 `mixed_dual_seed{42,43,44}`、`mixed_fixed_seed{42,43,44}` 和 `mixed_dual_denoised_seed{42,43,44}`，共 9 组。需要比较训练增强策略时，也可使用另外 6 组。

每组包含 `train_clean.npz` 与 `validation_{clean,motion,noise,dark,jpeg}.npz`。用 NumPy 的 `np.load(path, allow_pickle=False)` 打开。训练文件有 2,940 行，验证文件各有 840 行。每一行的 `features` 是融合后的 512 维表示，`raw_features` 和 `enhanced_features` 是同一图片的两路表示。`gate` 为形状 `(N, 1)` 的增强分支权重，固定平均融合恒为 0.5。`image`、`track_id`、`y_true`、`y_pred` 与特征行一一对应。

每组模型独立用 `train_clean.npz` 拟合 PCA/KMeans，然后使用同一套变换和聚类中心处理该模型的五个验证条件。按 `image` 配对并核对 `track_id` 和 `y_true`；标签仅用于计算 ARI、NMI 等评价指标。计算干扰前后的聚类一致率时，使用同一图片的配对。需要重采样时以轨迹为单位。

## 交接范围

包内保存 C 的正式开发验证产物。A 的 13 条件原评估实现及 A/B/C 的输入对齐仍需按全组协议确认。C 已用同一权重测出单路与双路的开销，拿到 A 的 `mixed_raw_seed42` 完整运行后可补做正式单分支的同机对照。最后的独立测试由全组冻结方案后统一进行。 [同机计时说明](../results/c_inference_cost_rtx4060ti_20261123/README.md)
