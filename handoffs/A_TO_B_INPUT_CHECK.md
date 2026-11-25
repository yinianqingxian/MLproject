# A → B：开发集输入对齐

> 2026-11-25 更新。A 与 C 的十三个条件像素指纹已全部一致；B 的 v2 材料只有 4 个条件一致。
> 本页说明 B 怎么在新机器上把输入对齐到可以配对的状态，以及 A 用哪两条命令验收。

## 1. B 需要准备什么

| 项目 | 具体要求 |
|---|---|
| 代码 | `git switch a/data-protocol-evaluation && git pull`，用**最新 commit**。旧版会把 `a_conditions.py`、`evaluate_a13.py` 的源码指纹记录成别的值 |
| 环境 | `opencv-python-headless==5.0.0.93`（cv2 报 5.0.0）、`numpy==2.4.1`。这是 A 的基准环境；版本号对上才可能逐像素一致 |
| 数据 | **不需要额外提供**。A 已核对：B 的 3,780 个源文件 sha256 逐条相同、1,470 个增强索引相同、增强数量相同、划分相同 |
| 权重 | `assets/weights/resnet18-f37072fd.pth`（不在仓库里，按 `assets/weights/README.md` 下载并校验 SHA256） |

环境装好后先自检：

```bash
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"   # 期望 5.0.0 2.4.1
```

## 2. 两条验收命令（都不需要 GPU、不需要权重）

**2.1 评估条件**——比较十三个条件的像素：

```bash
python experiments/evaluate_a13.py --prepare-only --device cpu --output results/b_input_check
python tools/compare_inputs.py --reference results/a_eval13_local_v1 \
  --candidate results/b_input_check --json results/b_input_diff.json
```

**2.2 训练输入**——比较三种策略与五个验证条件的生成像素：

```bash
python tools/compare_inputs.py --reference results/a_local_v1 \
  --candidate results/b_newpc_clahe --json results/b_train_diff.json
```

第二条命令要在 B 完成训练之后才能跑（它读 `results/b_newpc_clahe/data_manifest.json`）。
训练**之前**可以先只跑 2.1；三项全绿再开训，能省下 12 组白跑。

## 3. 判定标准

| 检查项 | 通过条件 | 不通过说明什么 |
|---|---|---|
| `split_sha256` | 相同 | 使用了不同的 `pilot_split.csv` |
| 图像清单 | 840 张，集合与顺序都相同 | 评估图不同，逐图对齐会错位 |
| 源图像哈希 | 3,780 条逐条相同 | GTSRB 原始文件不同源 |
| 增强索引 / 数量 | 相同 | 随机序列没有复现 |
| `data_fingerprint` | 相同 | 生成像素整体不同（OpenCV/NumPy 构建或代码版本不同） |
| 十三个 `pixel_sha256` | 13/13 相同 | 具体哪几个干扰条件不同，命令会列出来 |
| 八个 split 生成指纹 | 8/8 相同 | 哪个训练/验证 split 不同，命令会列出来 |

## 4. 已知的坑

1. **JPEG 与 dark 高强度"恰好一致"不代表对齐。** JPEG 量化和 ×0.25 截断会把 ±1 的底图差异吃掉。只看整体 `data_fingerprint` 和逐条件表，不要因为 4 个条件相同就认为部分对齐。
2. **同一版本号在不同平台仍可能差 1 个 LSB。** 若 B 在 macOS 上把 OpenCV 降到 5.0.0 后仍不一致，说明平台构建差异无法通过钉版本解决，只能改用官方测试做统一评估（A 在自己的环境里评估所有人的权重）。
3. **训练后再改 C 的五个核心文件会让 `--verify-only` 失败。** `run_resnet_study.py`、`resnet_robustness_study.py`、`run_c_study.py`、`cnn_robustness_study.py`、`cnn_robustness_pilot.py` 的内容指纹写进了运行配置，训练期间不要动它们。

## 5. A 不会改的东西

A 只提供基准清单与比对工具，不改 B 的评估目录、不改 B 的训练产物、不替 B 重跑。
指纹对不上的处理办法是全组统一环境后重跑，而不是修改指纹记录。
