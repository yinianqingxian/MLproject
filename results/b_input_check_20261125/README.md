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

差异模式指向**缩放插值**：不同平台的 OpenCV 在 `cv2.resize(..., INTER_AREA)` 上会产生少量像素的 ±1 差异，
之后所有基于干净图的干扰都被这个基础差异带偏。macOS arm64 与 Windows x64 即使版本号相同也不例外（见第 4 节实测）。

### 3.2 代码版本：指纹不同，但像素相同（已排除）

B 的 `conditions.json` 记录的源码指纹与 A 的基准不同（`a_conditions.py` `f10257f8…` vs `b0c970af…`），
且缺少 `scope` 字段，说明他用的是较早的代码修订。

**但这不是像素差异的原因。** A 把仓库里保留的旧版 `a_conditions.py`（来自早期训练包）与当前版本放在同一环境中对比，
十三个条件的指纹**逐位相同（13/13）**，且两者都与已验收基准一致。也就是说新版是对旧版的重构与扩展
（增加 `manifest()`、官方测试条件、`--prepare-only`、自检计数），**像素生成逻辑没有改动**。

结论：代码版本差异只影响溯源字段（`source_hashes`、`scope`），不影响像素。要如实记录，但不必为它重跑。

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

**已实测确认（2026-11-25）**：B 在 macOS 上把 OpenCV 降到 5.0.0（与 A 完全相同的版本号）后重跑，
`validation/clean` 仍是 `e389f327de3a`，与他在 5.1.0 下的结果逐位相同；而 A 在 Windows 5.0.0 上是
`3b71bffa7023`。结论：**差异来自 macOS arm64 与 Windows x64 的 OpenCV 构建（INTER_AREA 的 SIMD/定点实现），
与版本号无关，无法通过钉版本解决**。

因此路线 A 在 Mac 上不可行。要得到可配对的开发集结果，只有两条路：
把 B 的 12 组放到与 A 相同的 Windows 环境运行（4060 级显卡约 47 分钟），
或者接受开发集跨成员差值混入环境因素、把正式对照放到官方测试。

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
