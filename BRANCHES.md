# 分支与模型对照

> 2026-11-28 整理。基线：`main` @ `6840014`（PR #3 合并后）。

## 一、两个模型

| | 旧模型：全局门控 | 新模型：空间门控 |
|---|---|---|
| 融合方式 | 每张图一个权重 | 每个位置一个权重 |
| 代码位置 | `resnet_robustness_study.py` 的 `fusion="learned"` | 同文件 `fusion="spatial"` |
| 输入模式名 | `dual`、`dual_denoised` | `dual_spatial` |
| 门控模块 | `nn.Linear(1024, 1)` | `nn.Conv2d(1024, 1, 1)` |
| 门控参数量 | 1,025 | 1,025（相同） |
| 结果目录 | A：`results/a_local_v1`<br>B：`results/b_clahe`、`results/b_denoised_clahe`<br>C：`results/study_resnet18_rtx4060ti_v1` | `results/spatial_gate_study_v1` 等 |
| 所在分支 | `main`、`a/data-protocol-evaluation` | `a/spatial-gate-train` |

单视图（`raw` / `clahe` / `denoised_clahe`）没有门控，属于旧模型这一代的基础对照。

## 二、分支

| 分支 | 最新提交 | 相对 main | 内容 | 处置 |
|---|---|---|---|---|
| `main` | `6840014` | 基线 | 全组已合并内容 | 保留 |
| `a/spatial-gate-train` | `aff5ead` | +4 | **新模型**：空间门控实现、9 次训练、评估、干预、报告；另有 C 在 11-27 加的平均池化与聚类指南 | **保留，新模型主线** |
| `a/data-protocol-evaluation` | `a04508e` | +8 | **旧模型时代的数据协议**：B 的 12 组结果、输入对齐工具、代码等价性证明 | 合并进 main 后再删 |
| `c/resnet18-handoff-cost` | `5af2299` | +0 | C 的交接与推理成本记录 | 已全部合并，可删 |
| `docs/add-pr-guidance` | `cb666c7` | +1 | PR 模板与贡献指南 | 合并后再删 |
| 标签 `archive/stage1-probe` | `4d36794` | — | 第一阶段推理探针（依赖 C 的旧权重，已被第二阶段替代） | 存档，不占分支 |

## 三、命名规则

- `a/`、`c/`、`d/` 开头表示成员；`docs/` 表示公共文档。
- 新模型的工作一律放在 `a/spatial-gate-train`，不再新开分支。
- 已合并的分支合并后立即删除；需要留档的用 `archive/` 标签而不是分支。

## 四、常用命令

```bash
# 看每个分支还有多少没合并进 main 的提交
git log --oneline main..<branch>

# 确认某分支是否已完全合并（无输出即已合并）
git cherry main <branch>

# 删除已合并的本地/远程分支
git branch -d <branch>
git push origin --delete <branch>
```
