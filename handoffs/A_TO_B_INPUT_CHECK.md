# A → B：开发集输入对齐

> 2026-11-25。A 与 C 的十三个条件像素指纹已全部一致，B 现有材料只有 4 个条件一致。
> 本页说明 A 与 B 怎么把输入对齐到可以配对比较的状态。协议见 [A_PROTOCOL.md](../A_PROTOCOL.md)。

## 1. B 先交这些

| 文件 | 位置 | 用途 |
|---|---|---|
| `conditions.json` | B 的开发集评估目录（`evaluate_a13.py` 的输出） | 十三个条件的参数与像素指纹 |
| `images.csv` | 同上 | 840 张图的顺序、标签、轨迹 |
| `run_config.json` | B 的训练研究目录 | 训练时记录的环境与整体数据指纹 |
| `data_manifest.json` | 同上 | 训练集各策略的生成像素指纹 |

如果 B 只做过训练、没跑过十三条件评估，让他先在自己机器上生成清单（不需要 GPU、不需要权重）：

```bash
python experiments/evaluate_a13.py --prepare-only --output results/b_input_check_v1
```

## 2. A 跑一条命令比对

基准就是 PR #3 里已经验收的 A 目录，不需要重新生成：

```bash
python tools/compare_inputs.py \
  --reference results/a_eval13_local_v1 \
  --candidate b_交上来的目录 \
  --json results/b_input_diff.json
```

输出会逐项列出：划分指纹、图像集合与顺序、`data_fingerprint`、十三个条件各自的 `pixel_sha256`，
并在最后给出「可以直接配对」或「不能直接配对」的结论，同时以退出码 0/1 表示。

## 3. 判定标准

| 检查项 | 通过条件 | 不通过说明什么 |
|---|---|---|
| `split_sha256` | 两边相同 | 使用了不同的 `pilot_split.csv`，划分被改过 |
| 图像清单 | 840 张，集合与顺序都相同 | 评估图不同，逐图对齐会错位 |
| `data_fingerprint` | 相同 | 生成像素整体不同（OpenCV/NumPy 构建或代码版本不同） |
| 十三个 `pixel_sha256` | 13/13 相同 | 具体是哪些干扰条件不同 |

**注意区分两种对齐：**

- **评估输入对齐**：`conditions.json` 一致即可，B 的旧模型可以在同一批图上重评。
- **训练输入对齐**：还要比较 `run_config.json` / `data_manifest.json` 的指纹。如果训练时的像素本来就不同，
  只对齐评估输入只能让「评估图」一致，不能把 B 纳入严格的跨组对照——那种情况需要用对齐后的流程重跑受影响的 B 组。

## 4. A 不会改的东西

A 只提供基准清单与比对工具，不改 B 的评估目录、不改 B 的训练产物、不替 B 重跑。
若 B 的指纹因为依赖版本不同而对不上，处理办法是全组统一环境后重跑，而不是修改指纹记录。
