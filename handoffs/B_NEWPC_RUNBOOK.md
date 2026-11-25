# B 的 12 次训练：操作手册（GPU 或 Mac 都适用）

> 2026-11-25。先对齐输入，再训练 9 + 3 次，然后评估、比对、打包。
> 全部步骤由 `tools/run_b_pipeline.py` 串起来。`--device` 选 `cuda`（NVIDIA 机器）或 `cpu`（Apple Silicon）。

## 0. 前提

| 项目 | 要求 |
|---|---|
| 代码 | 含 `run_b_pipeline.py` 的版本（`0f3e584` 之后） |
| 环境 | `pip install -r requirements-b.txt`；`python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"` 必须是 **5.0.0 2.4.1** |
| torch | GPU 机器按 CUDA 装；Mac 直接 `pip install torch torchvision` |
| 数据 | 完整 GTSRB 放在 `data/GTSRB/raw/`（**不用重新下载**，A 已核对源文件逐条相同） |
| 权重 | `assets/weights/resnet18-f37072fd.pth`（不在仓库里，按 `assets/weights/README.md` 下载并核对 SHA256） |

## 1. 七个步骤

```bash
# NVIDIA 机器：--device 可省（会优先用 CUDA）
python tools/run_b_pipeline.py check
python tools/run_b_pipeline.py probe
python tools/run_b_pipeline.py inputs
python tools/run_b_pipeline.py train
python tools/run_b_pipeline.py eval
python tools/run_b_pipeline.py compare
python tools/run_b_pipeline.py package
```

Mac（Apple Silicon）把设备换成 cpu、线程调高：

```bash
python tools/run_b_pipeline.py check  --device cpu
python tools/run_b_pipeline.py probe  --device cpu --threads 10
python tools/run_b_pipeline.py inputs --device cpu
caffeinate -i python tools/run_b_pipeline.py train --device cpu --threads 10
python tools/run_b_pipeline.py eval   --device cpu
python tools/run_b_pipeline.py compare
python tools/run_b_pipeline.py package
```

`caffeinate -i` 防止训练途中睡眠；合盖仍会中断，训练时别合盖。

### 各步骤的门槛

| 步骤 | 通过条件 | 不通过怎么办 |
|---|---|---|
| `check` | cv2 5.0.0、numpy 2.4.1、图 39,209/12,630、划分与权重指纹一致 | 按提示补装或补文件 |
| `probe` | 只输出估算 | —— |
| `inputs` | `13/13 一致` | **默认停下**；跨平台（Mac）可用 `--allow-pixel-mismatch` 继续，见第 4 节 |
| `train` | 12 个运行都有 `completion.json` 且 `--verify-only` 通过 | 同命令加 `--resume` 补；半成品换新目录 |
| `eval` | 指标 156 行、逐图预测 131,040 条 | 数量不对不许打包 |
| `compare` | 训练 8/8、条件 13/13 | 有差异照样打包，但结论按实际写 |
| `package` | 生成 `b_newpc_results.zip` | —— |

## 2. 耗时

一次训练固定 350 次优化更新（4,410 ÷ 128 = 每轮 35 步 × 10 轮），总时间 ≈ 350 × 单步耗时 × 次数。
`probe` 会测出本机真实值，**以它为准**。参考：

| 设备 | 单步（batch 128） | 一次训练 | 12 次训练 | 两批评估 |
|---|---|---|---|---|
| RTX 4060 Laptop（实测） | 0.67–0.72 s | **3.9 分钟** | **约 47 分钟** | 约 8 分钟 |
| 桌面 4090/5090（估算） | 0.2–0.3 s | 1.2–1.8 分钟 | 15–22 分钟 | 3–5 分钟 |
| Kaggle T4（估算） | ~1.0 s | 约 6 分钟 | 约 72 分钟 | 约 12 分钟 |
| **M3 Max / Pro（估算）** | 3–6 s | 18–35 分钟 | **3.5–7 小时** | 0.5–1 小时 |
| x86 CPU 24 核（实测） | 4–5 s | 25–30 分钟 | 5–6 小时 | 约 40 分钟 |

结论：有 NVIDIA 卡就是 1 小时以内；只有 Mac 也能做完，**留一个晚上**。

## 3. 两批训练的配置

| 批次 | 输入模式 | 增强策略 | 种子 | 次数 | 输出目录 |
|---|---|---|---|---|---|
| 实验一 | `clahe` | `repeat_clean` / `targeted_motion` / `mixed` | 42、43、44 | 9 | `results/b_clahe` |
| 实验二 | `denoised_clahe` | 仅 `mixed` | 42、43、44 | 3 | `results/b_denoised_clahe` |

共用 A 的划分、4,410 条预算、官方 ResNet18 初始化、batch 128、10 轮、只评最后一轮。

## 4. 在 Mac 上训练要注意的三件事

1. **只能用 CPU。** 公共训练入口只接受 `cpu` / `cuda`；`mps` 要改 C 的冻结文件，会破坏全组同一套训练入口，也改不动已训练产物的指纹。
2. **像素很可能和 A 对不上。** OpenCV 5.1.0 与 5.0.0 在 `cv2.resize(INTER_AREA)` 上已有差异，macOS 与 Windows 即使版本号相同也可能差 1 个 LSB。所以 `inputs` 大概率不通过，用 `--allow-pixel-mismatch` 继续。
3. **这不影响你回答自己那两个问题。** 实验二（去噪 vs 不去噪）两组都在同一台 Mac 上训练，组内对照是干净的；实验一相对 A 的跨成员差值会混入环境因素，**统一改由官方测试评估**（A 在自己的环境里评估所有成员的末轮权重）。

## 5. 交付内容

`b_newpc_results.zip` 里包含：

| 内容 | 用途 |
|---|---|
| 两个训练目录（`run_config.json`、`data_manifest.json`、`runs/*/model.pt`、`completion.json`） | A 核对输入、冻结方案后统一跑官方测试 |
| 两批评估（`metrics.csv`、`predictions.csv`、`costs.csv`、`conditions.json`） | D 复算与统计 |
| 四份比对报告（`results/b_*_diff.json`） | 证明输入是否对齐 |
| `results/b_input_check/`（条件清单） | 评估输入的证据 |

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `inputs` 报 N 个条件不一致 | 版本或平台差异 | GPU 机器把 cv2/numpy 降到 5.0.0/2.4.1 重试；Mac 用 `--allow-pixel-mismatch` |
| 只有 JPEG 和 dark 高强度一致 | 量化把微小差异吃了，**不是**部分对齐 | 看整体 `data_fingerprint`，别看个别条件 |
| `Study code fingerprints differ` | 训练后改过 `experiments/` 下的核心文件 | 换新输出目录重跑 |
| 训练中断 | —— | 同命令加 `--resume`，只补未完成的运行 |
| 评估报 `trained on different generated pixels` | 训练和评估不在同一环境 | 在同一台机器、同一环境里评估 |
| Mac 上风扇狂转、系统卡 | 12 次训练是满载 CPU | 正常；建议插电源、留出散热空间 |
