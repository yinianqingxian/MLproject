# C 部分结果汇总与复现

状态：历史 SmallCNN 结果核验与报告生成说明。本仓库保留汇总、图表与验收记录。以下重建报告和核验命令需要本地完整的 `results/study_v1` 运行产物。面向接手 C 产物的小组成员；本目录仅覆盖开发验证，官方测试及 A/B 正式单分支结果不在本次交付范围。

在项目根目录使用本项目 Python 环境运行：

```powershell
.\.venv\Scripts\python.exe .\experiments\summarize_c_study.py --study-dir "results/study_v1" --report-dir "report/c_part"
```

运行前必须已有完整的 15 次训练及 `cnn_robustness_study.py`；本命令不训练模型。依赖为项目环境内的 NumPy、Matplotlib 及训练模块所需依赖，固定版本见项目 `requirements.lock`。源配置中环境记录的不同取值为：

```json
[
  {
    "cpu_count": 16,
    "cuda_device": null,
    "cuda_runtime": null,
    "device": "cpu",
    "numpy": "2.1.2",
    "opencv": "4.10.0",
    "platform": "Windows-11-10.0.26200-SP0",
    "processor": "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel",
    "python": "3.13.0",
    "sklearn": "1.8.0",
    "torch": "2.15.0+cpu"
  }
]
```

环境记录缺失时，不从当前机器推断训练时版本；应直接检查每次运行的 `config.json` 与项目运行配置。硬件及线程相同是比较墙钟成本的前提，跨机器运行不应仅据原始耗时排序。

完整重训入口为 `experiments/run_c_study.py`，默认读取 `data/GTSRB/raw` 和 `results/pilot_split.csv`，不读取官方测试集。以下命令采用本次记录的设备与线程，并写入新的复现目录；该目录必须尚不存在：

```powershell
.\.venv\Scripts\python.exe .\experiments\run_c_study.py --output-dir "results/study_v1_reproduction" --epochs 10 --threads 6 --device cpu
```

本次设备记录为 `cpu`，线程记录为 `6`；若有多个值，应按各运行配置分别复现，不把上述第一种环境当成所有运行的环境。更改线程、硬件或依赖可能影响时间及浮点结果，复现时须记录环境变化。新目录生成 15 个完成运行后，再把第一条汇总命令的 `--study-dir` 改为该目录，并把 `--report-dir` 改为独立目录以保留两次报告。

仅校验已有研究目录的训练产物可运行：

```powershell
.\.venv\Scripts\python.exe .\experiments\run_c_study.py --output-dir "results/study_v1" --verify-only
```

训练中断时使用同一训练命令加 `--resume`；它只在冻结配置、代码、环境和完成产物匹配时跳过已完成运行，遇到半成品仍停止，不会静默覆盖。应保留失败目录并由 C 定位问题，再在新输出目录重跑。`--verify-only` 检查冻结配置与产物指纹；汇总程序另外从逐图预测独立复算指标。

## 输入与验收

五种配置为 dual × repeat_clean/targeted_motion/mixed，以及 mixed × fixed/dual_denoised；每种种子 42、43、44。`runs/<run_id>/` 必须包含 config.json、model.pt、训练模块完整性记录、training_log.csv、metrics.csv、predictions.csv、cost.json 和六个 `features/*.npz`。learned 模型还需 intervention_metrics.csv 与 intervention_predictions.csv；fixed 不提供门控干预。划分优先读取研究目录的 split.csv 或 pilot_split.csv，否则读取项目 results/pilot_split.csv。

| 产物 | 每运行数量 / 核验要点 |
| --- | --- |
| metrics.csv | 5 条；clean、motion、noise、dark、jpeg；固定七类重算 F1/准确率 |
| predictions.csv | 4,200 条；每条件同一批 840 张、28 条轨迹 |
| training_log.csv | 10 轮；末轮 clean 分数与指标一致 |
| features/train_clean.npz | 2,940 张、98 条轨迹、32 维、按图像名排序 |
| features/validation_*.npz | 5 × 840 张；图像、轨迹、真值、预测与 CSV 对齐 |
| intervention_metrics.csv | learned 模型 15 条；固定 g=0、0.5、1 |
| intervention_predictions.csv | learned 模型 12,600 条；干预指标独立重算 |
| cost.json | 参数、训练/导出/各条件前向时间、4,410 条目、350 步、线程 |

全部成功应显示 15 次运行、75 条常规指标和 63,000 条常规预测，并生成 `summary_manifest.json`。样本标准差使用 ddof=1，配对差值使用同种子模型逐一相减。帧级门控分布是描述性统计，不用于逐帧 IID 置信区间。

## 输出

- 研究目录根部的 metrics.csv、predictions.csv、intervention_metrics.csv、intervention_predictions.csv：核验后的总表。
- `tables/metric_summary.csv`：均值与样本 SD；`paired_by_seed.csv` 与 `paired_summary.csv`：五种配对比较。
- `tables/gate_distribution.csv`：每运行、训练/验证条件、整体与类别的门控分布、图像和独立轨迹数。
- `tables/intervention_by_run.csv` 与 `intervention_summary.csv`：同权重干预，与独立训练 fixed 分开。
- `tables/training_log.csv`、`convergence.csv`、`cost_by_run.csv`、`cost_summary.csv`、`inference_cost.csv`：训练轨迹及成本。
- `figures/`：scores_with_sd, paired_changes, gate_distribution, training_curves, intervention_effects，各有 PNG 与 SVG；图用英文以避免中文字体缺失。
- `C_REPORT.md`：中文模型、结果、分析与边界；`summary_manifest.json`：源文件哈希和核验计数。

汇总会重建这些派生文件；每次运行目录和预实验材料只读保留。先通过所有输入核验才开始写输出，缺少运行或半成品会以非零状态退出，并打印首个错误，不生成部分矩阵均值。绘图期间异常时，已有输入不受影响；修复后用同一命令重建派生文件，并以新的 summary_manifest.json 为完整生成的完成标志。

## 常见失败与交接

- 数量/配置不符：保留错误运行，回到训练入口核验完成状态与冻结配置；不要手改 CSV 或删除完整性记录来绕过验证。
- 指纹或重算指标不符：停止使用对应产物，由 C 检查运行文件是否修改以及特征/预测导出是否正确，重新在独立运行目录训练。
- 门控或元数据不对齐：检查图像排序、类别原始编号及轨迹标识；不能通过重新排序标签掩盖问题。
- 需要置信区间：交给 D 按整条轨迹实施配对 bootstrap，保留同图像的跨条件配对。当前三种子 SD 不是置信区间。
- 正式单分支、聚类、多强度和独立测试由 A/B/D 与全组后续完成；不能将端点干预当成独立训练分支。

协议来源为项目 HANDOFF_C.md、02-c-implementation.md 和 01-project-plan.md 的成员 C 与固定协议章节。结果解释见 [C_REPORT.md](C_REPORT.md)，所有表格可追溯至运行 ID、种子和原始预测。
