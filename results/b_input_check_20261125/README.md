# B 开发集输入核对结果（v2）

> 2026-11-25。对象：`给A_B输入核对_v2/B_input_check_v2`（B 提交，含 `clahe/` 9 次与 `denoised_clahe/` 3 次两批）。
> 基准：`results/a_eval13_local_v1/`（PR #3 已验收，A/C 十三个条件指纹全部一致）。

## 1. 完整性

`清单.sha256` 的 10 个文件全部校验通过，包没有损坏或缺失。

## 2. 结论：**不能直接配对**

两批（`clahe`、`denoised_clahe`）结果一致：**13 个条件里有 9 个像素不一致**，整体 `data_fingerprint` 不同。

| 检查项 | 结果 |
|---|---|
| `split_sha256` | **一致**（`de914a91…`，同一份冻结划分） |
| 图像清单 | **一致**（840 张，集合与顺序都相同） |
| `data_fingerprint` | **不同**：A `e494bcb9…` / B `4f3a1790…` |
| 条件像素 | **4 / 13 一致** |

逐条件：

| 条件 | 一致 | 条件 | 一致 |
|---|---|---|---|
| clean | ✗ | dark 低 | ✗ |
| motion 低/中/高 | ✗ ✗ ✗ | dark 中 | ✗ |
| noise 低/中/高 | ✗ ✗ ✗ | **dark 高** | **✓** |
| **jpeg 低/中/高** | **✓ ✓ ✓** | | |

训练集的三个策略指纹也全部不同：`training/repeat_clean`、`training/targeted_motion`、`training/mixed`。

## 3. 原因

### 3.1 环境不同（主因）

| | A | B |
|---|---|---|
| 平台 | Windows-11 x86_64 | macOS-26.6.2 arm64 |
| Python | 3.14.2 | 3.11.15 |
| OpenCV | **5.0.0** | **5.1.0** |
| NumPy | **2.4.1** | **2.4.6** |
| scikit-learn | 1.8.0 | 1.9.1 |
| torch | 2.14.0+cu130 | 2.15.0 |

差异模式指向**缩放插值**：`cv2.resize(..., INTER_AREA)` 在 OpenCV 5.0.0 与 5.1.0 之间会产生少量像素的 ±1 差异，
之后所有基于干净图的干扰都被这个基础差异带偏。

### 3.2 代码版本不同（次要，但必须消除）

`conditions.json` 里记录的源码指纹与 A 的基准不同：

| 文件 | A（基准） | B |
|---|---|---|
| `a_conditions.py` | `b0c970af…` | `f10257f8…` |
| `evaluate_a13.py` | `6d36122d…` | `39bc14a4…` |

B 的 `conditions.json` 还缺少 `scope` 字段，说明它是**用较早或不同的代码版本**生成的。
这意味着即使环境对上，也不能保证 13 个条件的生成逻辑完全相同。A 当前文件的指纹与基准记录一致，基准本身没有漂移。

### 3.3 为什么 JPEG 和 dark 高"恰好一致"

这两类一致**不代表对齐**，是量化把微小差异吸收了：

- JPEG（q=40/20/10）量化很重，干净图上 ±1 的差异大多在 DCT 量化后消失；
- dark ×0.25 后转 `uint8` 截断，`200×0.25=50` 与 `201×0.25=50.25→50` 落到同一个值。

而 ×0.7、×0.45、模糊（邻域平均会扩散差异）、噪声（叠加在已不同的底图上）都保留了差异。
所以 4/13 是巧合，**不能**当作"部分条件已经对齐"。

## 4. 两条处置路线

### 路线 A：让 B 对齐到基准环境（可以配对，但要重训）

1. 用同一个 commit（`8d40ddf` 或更新）的代码；
2. OpenCV 装 5.0.0（`pip install opencv-python-headless==5.0.0.93`），NumPy 尽量 2.4.1；
3. 重新 `python experiments/evaluate_a13.py --prepare-only --device cpu --output <新目录>`；
4. 再跑一次 `tools/compare_inputs.py`。

若这样仍不一致，说明同版本号在不同平台上的构建仍有差异，Mac 与 Windows 无法逐像素对齐，
那就只能采用路线 B，或把 B 的 12 组在 A 的环境里重训。

### 路线 B：不做开发集跨成员配对，把正式对照放到官方测试（不需要 B 重训）

A 在自己环境里统一评估所有成员的**末轮权重**：测试像素由 A 生成，所有模型看到完全相同的测试图。
这也是 A 负责的官方七类测试的原定做法。前提是 B 交出**完整的 runs 目录**（含 `model.pt`、`features/*.npz`、`completion.json`），
不能只交 `metrics.csv`。

## 5. 对 D 的方法学提醒

只要 B 的训练像素与 A 不同，**跨成员差值就同时混入了"输入方式"和"环境"两个因素**，
不能用来解释 CLAHE 或增强策略本身的效应。可以成立的是：

- **组内比较**（B 自己的三种增强、A 自己的三种增强、C 自己的融合对照）——同一环境内自洽；
- **官方测试上的模型对比**——测试图统一，但结论要写明各模型的训练环境不同。

## 6. 复现命令

```bash
python tools/compare_inputs.py --reference results/a_eval13_local_v1 \
  --candidate <B_input_check_v2>/clahe --json results/b_input_check_20261125/clahe_diff.json
python tools/compare_inputs.py --reference results/a_eval13_local_v1 \
  --candidate <B_input_check_v2>/denoised_clahe --json results/b_input_check_20261125/denoised_clahe_diff.json
```
