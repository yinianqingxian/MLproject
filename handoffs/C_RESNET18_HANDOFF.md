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

原交接 ZIP 保存 C 的训练与五条件开发验证产物。C 的 15 份末轮权重已按 A 的评估协议补齐十三条件推理，共有 195 行指标、163,800 条逐图预测和 195 份验证特征。十三条件特征保存在 C 的评估原始目录，可与 ZIP 中同一 `run_id` 的 `train_clean.npz` 配对。 [十三条件报告](../report/c_part_eval13_rtx4060ti/C_REPORT.md)

A 的正式结果和权重已收到并核验。A 与 C 的 840 张验证图、划分和 13 个条件像素指纹一致。B 的现有材料有 4 个条件与 A/C 的像素指纹一致，取得 B 的完整产物后再复核训练数据指纹和输入视图。官方七类独立测试由全组冻结方案后统一进行。 [A 的交接说明](A_TO_D_HANDOFF.md) · [A 的条件清单](../results/a_eval13_local_v1/conditions.json) · [C 的条件清单](../results/c_eval13_rtx4060ti_v1/conditions.json)

## A 与 C 的同机计时

正式对照已复测完成。RTX 4060 Ti 上，A 的原图单路平均为 **0.70–0.72 毫秒/图**，C 的门控双路为 **1.44–1.45 毫秒/图**，五个开发条件的配对倍率为 **2.01–2.06 倍**。按 batch 128 的吞吐折算，视图缓存构建耗时单独记录。仓库附有 45 条原始计时、10 条汇总和两份末轮权重的 SHA-256。A/C 正式对照取 `summary.csv` 中 `single_input=raw` 的五行。 [原始计时](../results/c_inference_cost_a_vs_c_rtx4060ti_20261124_recheck/timings.csv) · [配对汇总](../results/c_inference_cost_a_vs_c_rtx4060ti_20261124_recheck/summary.csv) · [环境及指纹](../results/c_inference_cost_a_vs_c_rtx4060ti_20261124_recheck/benchmark_manifest.json)

复测时在仓库根目录准备好 GTSRB 原图、`results/pilot_split.csv` 和 A、C 两组完整运行目录。以下命令写入新的结果目录。脚本会先核验权重、运行产物、种子、训练预算和数据指纹。 [数据目录说明](../README.md)

```bash
python -u tools/benchmark_resnet_inference.py \
  --raw-run results/a_local_v1/runs/mixed_raw_seed42 \
  --fusion-run results/study_resnet18_rtx4060ti_v1/runs/mixed_dual_seed42 \
  --output results/ac_inference_cost_recheck
```

## 复跑十三条件评估

在仓库根目录准备好 GTSRB 原图和完整的 C 训练目录，使用与训练数据指纹一致的 Python 环境。下面的命令从已验收权重推理，输出目录须为新目录。 [数据目录说明](../README.md)

```bash
NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python experiments/evaluate_a13.py \
  --study results/study_resnet18_rtx4060ti_v1 --expect-runs 15 --device cuda \
  --data-root data/GTSRB/raw --split results/study_resnet18_rtx4060ti_v1/split.csv \
  --output results/c_eval13_rtx4060ti_recheck
```

以输出目录的 `completion.json` 和 `15 / 195 / 163800 / 195` 四项总数验收，分别对应模型、指标、逐图预测和特征文件。若评估发现数据指纹不同，先核对划分、OpenCV 和 NumPy 环境，再在与训练像素匹配的环境复跑。 [A 的十三条件字段约定](A_TO_D_HANDOFF.md)
