# A → D 交接说明

> 2026-11-23。A 的协议、代码和审计表已就绪；**九次训练与十三条件评估要在 A 的 Linux GPU 服务器上运行**，产物目录名固定，D 可以按本文的 schema 直接写合并与统计程序。冻结参数见 [A_PROTOCOL.md](../A_PROTOCOL.md)。

## 1. 交接范围

| 内容 | 状态 | 位置 |
|---|---|---|
| 数据/划分/预算审计 | 已完成 | `results/a_data_audit/` |
| 干扰与评估模块 | 已完成并通过回归测试 | `experiments/a_conditions.py`、`experiments/evaluate_a13.py`、`experiments/evaluate_official_test.py` |
| 汇总程序（表格/配对/曲线） | 已完成 | `experiments/summarize_a_study.py` → `report/a_part/` |
| 九次原图单分支训练 | **待 Linux 运行** | `results/a_resnet18_linux_v1/` |
| 开发集十三条件评估 | **待训练完成后运行** | `results/a_eval13_linux_v1/` |
| 官方七类测试评估 | 需全组冻结后运行 | `results/a_test7_v1/` |

D 现在就可以用审计表和 schema 写合并程序；训练产物到位后不需要改接口。

## 2. 目录结构

```text
results/a_resnet18_linux_v1/            九次训练（run_resnet_study.py 输出）
├── run_config.json                     {"common": {...}, "matrix": [[aug, mode, seed], ...]}
├── split.csv                           冻结划分副本
├── data_manifest.json                  数据生成指纹
└── runs/<augmentation>_<mode>_seed<seed>/
    ├── config.json  status.json  completion.json  cost.json
    ├── model.pt  training_log.csv
    ├── metrics.csv  predictions.csv          5 个开发验证条件
    └── features/{train_clean,validation_clean,validation_motion,
                  validation_noise,validation_dark,validation_jpeg}.npz

results/a_eval13_linux_v1/              十三条件评估（evaluate_a13.py 输出）
├── conditions.json  images.csv  completion.json
└── <同一个 run_id>/
    ├── metrics.csv  predictions.csv  costs.csv
    └── <condition>_<severity>.npz            13 份特征

results/a_test7_v1/                     官方测试（evaluate_official_test.py 输出）
├── conditions.json  test_manifest.csv  completion.json
└── <study 名>/<run_id>/                   同样的三类 csv + 13 份 npz
```

## 3. 字段契约

`metrics.csv`（每行一个条件）：

```text
run_id, seed, augmentation, input, split, condition, severity,
macro_f1, accuracy, n_images, parameters
```

`macro_f1` / `accuracy` 是 0–1 的小数，不是百分比；`split ∈ {train, validation, test}`；`severity ∈ {none, low, medium, high}`。

`predictions.csv`（每行一张图）：

```text
run_id, seed, split, image, track_id, condition, severity, y_true, y_pred, gate
```

`image` 是相对 `data/GTSRB/raw/` 的路径；单分支模型的 `gate` 为空字符串（不是 `nan`）；官方测试集的 `track_id` 为空字符串。

`costs.csv`（仅十三条件评估与官方测试评估）：

```text
condition, severity, parameter, images, batch_size, preprocessing_seconds, evaluation_seconds
```

特征 `.npz`（`np.load` 后是数组）：`features`（840×512 或 4110×512）、`y_pred`、`image`、`track_id`、`y_true`、`gate`；融合模型另有 `raw_features`、`enhanced_features`。**行已按 `image` 升序排好**，可以直接与 `predictions.csv` 按 `image` 对齐。

`conditions.json`：`levels`、`noise_seed`、`data_fingerprint`、`split_sha256`、`freeze_sha256`、`conditions[]`（每项含 `condition`、`severity`、`parameter`、`n_images`、`pixel_sha256`）、`source_hashes`。

`completion.json`：`runs[]`（`run_id`、`checkpoint_sha256` 等）、`totals`、`expected_totals`、`environment`、`artifact_hashes`。

## 4. 数量验收清单

跑完训练与评估后，D 应能核出以下数字；多一个或少一个都说明产物不完整。

| 检查项 | 期望值 |
|---|---|
| 训练运行目录 | 9 |
| 训练常规指标行（5 条件 × 9） | 45 |
| 训练逐图预测行（4,200 × 9） | 37,800 |
| 训练特征文件（6 × 9） | 54 |
| 评估指标行（13 × 9） | 117 |
| 评估逐图预测行（13 × 840 × 9） | 98,280 |
| 评估特征文件（13 × 9） | 117 |
| 每个配置的种子数 | 3（42/43/44） |

## 5. D 可以直接拿来用的汇总

`experiments/summarize_a_study.py` 已经输出：

- `tables/dev13_by_run.csv`：逐运行、逐条件的 Macro-F1/accuracy。
- `tables/dev13_summary.csv`：三种子均值 ± 样本 SD。
- `tables/dev13_paired.csv`：A1−A0、A2−A0、A2−A1 的**逐种子配对差值**，含每个种子的原始差值和“同向种子”计数。
- `tables/dev13_families.csv`：每个干扰族三档均值、12 干扰条件均值、最差条件。
- `tables/dev13_per_class.csv`：逐类 F1 的三种子均值 ± SD。
- `tables/dev13_costs.csv`：逐条件预处理与推理耗时。
- `figures/dev13_curves.png`、`dev13_conditions.png`、`dev13_families.png`。

这些是描述性汇总。**统计推断仍由 D 负责**：按开发集轨迹做配对 bootstrap、给出区间、判断是否达到预登记阈值。

## 6. D 需要自己做的事

1. **合并 A/B/C**：三者 schema 相同，按 `run_id` 前缀区分成员；注意 B/C 的 `input` 字段是 `clahe`/`dual` 等，A 是 `raw`。
2. **独立复算**：抽若干运行，用 `predictions.csv` 重算 Macro-F1 并与 `metrics.csv` 比对；这是 A 的验收要求。
3. **配对 bootstrap**：以 `track_id` 为单位重采样 2,000 次，同一次重采样同时用于两个模型，再算均值差的区间。不要把 840 帧当独立样本。
4. **聚类分析**：用 `features/*.npz` 的 512 维特征，在干净训练特征上拟合 KMeans（K=7、n_init=20、random_state=42），投影验证特征，算 ARI/NMI 与干净↔受损的分配一致率。
5. **跨环境核对**：对照 `data_fingerprint` 和每个条件的 `pixel_sha256`。若 A 的 Linux 环境与 B/C 的环境指纹不同，跨成员差值不能直接解释为方法差异，必须写进局限。

## 7. A 已固定、不应被 D 改动的部分

- 类别顺序 `1,2,3,4,5,7,8` 与索引 0–6。
- 13 个条件的参数、噪声种子（开发集 20261117，测试 20261111）与像素指纹。
- 三种增强策略的 4,410 条预算与 350 次更新。
- 只评最后一轮的规则。
- 官方测试的冻结流程：先写清单，再看分数。

若发现协议本身有问题，应作为新版本记录并**全组重跑**，不能在分析阶段就地调整。

## 8. 已知风险

| 风险 | 影响 | 处理 |
|---|---|---|
| Linux 环境与 Windows 像素指纹不同 | 跨成员配对可比性下降 | 报告局限中说明；必要时全组在同一机器重跑 |
| 开发集只有 28 条验证轨迹 | 区间宽、结论弱 | 按轨迹 bootstrap，报告区间而不只报点估计 |
| 官方测试集无轨迹 ID | 无法做轨迹级不确定性 | 只报逐图/逐类指标并注明 |
| A2 的模糊样本只有 A1 的 1/3 | A1→A2 比较混杂预算变化 | 结论中区分“混合的收益”与“针对性样本减少的代价” |
| 训练只有 10 轮 | 可能未收敛 | 报告训练曲线，说明这是短程开发验证 |

## 9. D 独立核验的命令

```bash
# 数据预算与轨迹交叉（不需要 GPU）
python tools/audit_a_data.py --output /tmp/a_audit && cat /tmp/a_audit/audit.json

# 只读核验九个训练运行与全部产物指纹
python experiments/run_resnet_study.py --inputs raw \
  --augmentations repeat_clean targeted_motion mixed --seeds 42 43 44 \
  --output-dir results/a_resnet18_linux_v1 --verify-only

# 复算十三条件汇总
python experiments/summarize_a_study.py --eval results/a_eval13_linux_v1 --output /tmp/a_report
```

有问题直接找我核对，不要自行修改 A 的产物目录。
