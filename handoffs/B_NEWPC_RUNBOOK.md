# B 在新机器上的 12 次训练：操作手册

> 2026-11-25。给 B 用：先对齐输入，再训练 9 + 3 次，然后评估、比对、打包。
> 全部步骤由一个脚本串起来：`tools/run_b_pipeline.py`。本文说明每一步的门槛和预计耗时。

## 0. 前提

| 项目 | 要求 |
|---|---|
| 代码 | `git switch a/data-protocol-evaluation && git pull`（含 `fa7ba3d` 之后的版本） |
| 环境 | `pip install -r requirements-b.txt`；`python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"` 必须打印 **5.0.0 2.4.1** |
| torch | 按机器的 CUDA 装（只影响速度与训练数值细节，不影响 32×32 像素） |
| 数据 | 完整 GTSRB 放在 `data/GTSRB/raw/`（**不用重新下载**，A 已核对源文件逐条相同） |
| 权重 | `assets/weights/resnet18-f37072fd.pth`（不在仓库里，按 `assets/weights/README.md` 下载并核对 SHA256） |

## 1. 六个步骤

```bash
python tools/run_b_pipeline.py check     # 环境、数据、划分、权重、版本
python tools/run_b_pipeline.py probe     # 测速，估算本机需要多久（不跑正式训练）
python tools/run_b_pipeline.py inputs    # 生成开发集条件清单，13/13 才放行
python tools/run_b_pipeline.py train     # clahe 9 次 + denoised_clahe 3 次
python tools/run_b_pipeline.py eval      # 两批十三条件评估
python tools/run_b_pipeline.py compare   # 训练像素 8/8、评估条件 13/13
python tools/run_b_pipeline.py package   # 打成 b_newpc_results.zip
```

也可以用 `python tools/run_b_pipeline.py all` 一次跑完；中途失败会停在那一步，不会带着不一致的输入继续。

### 各步骤的门槛

| 步骤 | 通过条件 | 不通过怎么办 |
|---|---|---|
| `check` | cv2 5.0.0、numpy 2.4.1、图 39,209/12,630、划分与权重指纹一致 | 按提示补装或补文件 |
| `probe` | 只输出估算，不做判断 | —— |
| `inputs` | `13/13 一致` | **停**，把比对报告给 A，不要开训 |
| `train` | 12 个运行全部 `completion.json` + `--verify-only` 通过 | 用 `--resume` 补；半成品换新目录 |
| `eval` | 指标 156 行、逐图预测 131,040 条 | 数量不对不要打包 |
| `compare` | 训练 8/8、条件 13/13 | 有差异照样打包，但结论要按实际写 |
| `package` | 生成 `b_newpc_results.zip` | —— |

## 2. 耗时估计

一次训练固定 350 次优化更新（4,410 条 ÷ batch 128 = 每轮 35 步 × 10 轮）。所以总时间 ≈ 350 × 单步耗时 × 次数。
`probe` 会直接测出本机的单步耗时并换算，**以它的输出为准**。下面是已实测的参考：

| 设备 | 单步（batch 128） | 一次训练 | 12 次训练 | 两批评估 |
|---|---|---|---|---|
| RTX 4060 Laptop（A 实测） | 0.67–0.72 s | **3.9 分钟** | **约 47 分钟** | 约 8 分钟 |
| 桌面 4090 / 5090 级（估算） | 0.2–0.3 s | 1.2–1.8 分钟 | 15–22 分钟 | 3–5 分钟 |
| Kaggle T4（估算） | 约 1.0 s | 约 6 分钟 | 约 72 分钟 | 约 12 分钟 |
| Mac / x86 CPU（实测 x86 24 核） | 3.9 s | 约 23 分钟 | 约 4.6 小时 | 约 1 小时 |

**总量级**：有一张 4060 级显卡时，12 次训练加评估大约 **1 小时**；没有 GPU 用 CPU 也只要 **5–6 小时**，但会很热。

## 3. 两批训练的配置

| 批次 | 输入模式 | 增强策略 | 种子 | 次数 | 输出目录 |
|---|---|---|---|---|---|
| 实验一 | `clahe` | `repeat_clean` / `targeted_motion` / `mixed` | 42、43、44 | 9 | `results/b_clahe` |
| 实验二 | `denoised_clahe` | 仅 `mixed` | 42、43、44 | 3 | `results/b_denoised_clahe` |

两组共用 A 的划分、4,410 条预算、官方 ResNet18 初始化、batch 128、10 轮、只评最后一轮。

## 4. 交付内容

`b_newpc_results.zip` 里包含：

| 内容 | 用途 |
|---|---|
| 两个训练目录（`run_config.json`、`data_manifest.json`、`runs/*/model.pt`、`completion.json`） | A 核对输入、冻结方案后统一跑官方测试 |
| 两批评估（`metrics.csv`、`predictions.csv`、`costs.csv`、`conditions.json`） | D 复算与统计 |
| 四份比对报告（`results/b_*_diff.json`） | 证明输入是否对齐 |
| `results/b_input_check/`（条件清单） | 评估输入的证据 |

## 5. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `inputs` 报 13 个条件里有 N 个不一致 | cv2/numpy 版本或代码版本不对 | 版本降到 5.0.0 / 2.4.1，用最新 commit 重新生成 |
| 只有 JPEG 和 dark 高强度一致 | 量化把微小差异吃了，**不是**部分对齐 | 看整体 `data_fingerprint`，不要看个别条件 |
| `Study code fingerprints differ` | 训练后改过 `experiments/` 下的核心文件 | 换新输出目录重跑 |
| 训练中断 | —— | 同一条命令加 `--resume`，只补未完成的运行 |
| 评估报 `trained on different generated pixels` | 训练和评估不在同一环境 | 在同一台机器、同一环境里评估 |
