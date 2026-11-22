# B 到 C 交接审查与下一步

本次已接收 B 的增强视图与单分支材料，并用 C 的公共预处理实际核对输入。B 的汇总结构齐全、模型协议与 C 一致，但跨环境输入像素尚未对齐，因此当前材料不能直接用于严格的 B/C 配对比较。C 的正式 ResNet18 15 次结果在已搜索的本机目录和相关压缩包中尚未找到。

## 已完成的推进

1. 原样接收 35 个交接文件并逐项核对 SHA256。仓库保留 [A 协议](../../handoffs/A_13_CONDITIONS_20261122.md) 和衍生审计，完整交接快照由小组保管。
2. 核验 12 份模型配置、156 条十三条件指标与 156 条计时记录，核对固定划分、官方初始化、预算和源码口径。配置声明与真实训练完成是两种证据，当前只有前者及汇总。
3. 用现有公共 `prepare_data` 和 `tensor_views` 重建五个已有开发条件，核验三视图与单分支/融合取图的一致性；本次没有训练或模型前向。
4. 新增可重复执行的标准库 [交接审计工具](../../tools/audit_b_handoff.py)，输出 [结构审计](summary/audit.json)、[B 三种子汇总](summary/b_metric_summary.csv)、[mixed 去噪逐种子配对](summary/b_mixed_denoising_pairs.csv)和 [去噪差值汇总](summary/b_mixed_denoising_summary.csv)。这些统计只涉及 B 自身，不包含 C 成绩。
5. 搜索可访问的本机项目与压缩包。旧模型和现有 Spark 训练 ZIP 均保留。

## B 交付中可以确认什么

| 项目 | 核验结果 |
| --- | --- |
| 汇总矩阵 | CLAHE 三策略 × 三种子 9 组，mixed 去噪 CLAHE × 三种子 3 组，共 12 组 |
| 开发集指标 | 每组 13 条件，共 156 行；每行 840 张图；分数标度为 0–1 |
| 配置声明 | 官方 ResNet18、ImageNet V1、512 维、10 轮、batch 128、AdamW 学习率 0.003、4,410 训练条目 |
| 官方初始化 | 12 份配置的完整权重 SHA256 均与当前项目一致 |
| 训练源码 | 五份声明指纹与当前项目一致；提供的视图模块逐字节相同 |
| 固定划分 | 与当前 `pilot_split.csv` 相同：2,940 训练、840 验证，98/28 条轨迹无交叉 |
| 训练完成证据 | 未提供模型、逐图预测、特征、逐轮训练日志和完成清单，暂不能重载模型、复算 Macro-F1 或完整验收训练预算 |

按 B 自己的表计算，mixed 下去噪 CLAHE 的三种子平均 Macro-F1 在 13 条件均低于 CLAHE，其中 clean 差约 −2.19 个百分点、motion medium 约 −9.20、dark high 约 −16.27。这是 B 汇总表呈现的均值趋势，不是 C 融合结论，也没有轨迹 bootstrap 或独立测试证据。

## 输入像素问题已经定位到哪一步

B 的 `输入核对_Mac.json` 与其 `conditions.json` 的本地十三条件指纹一致。但相对该 JSON 保存的 A 历史参考，只有 `dark/high` 与 JPEG 三档相同，其余 9 项不同。单靠使用相同干扰参数和文件名，不能确认实际输入一致。

本机使用 OpenCV 4.10.0、NumPy 1.26.4、Torch 2.5.1，实际按 B 脚本的连续 NCHW float32 字节口径计算视图 SHA256，结果为：

| 当前公共开发条件 | raw 对 B | CLAHE 对 B | 去噪 CLAHE 对 B |
| --- | --- | --- | --- |
| clean | 不同 | 不同 | 不同 |
| motion medium | 不同 | 不同 | 不同 |
| noise medium | 不同 | 不同 | 不同 |
| dark medium | 不同 | 不同 | 不同 |
| jpeg medium | 相同 | 相同 | 不同 |

完整记录见 [本机五条件三视图核验](local_five_condition_views.json)。本机所有对应单分支/融合输入均逐像素相同，说明共享取图实现配对正常；跨环境 15 项中只有 2 项匹配 B。B 报告的环境是 OpenCV 5.1.0、NumPy 2.4.6、Torch 2.15.0。当前证据不足以把差异全部归因于某一个库或某一项变换。

另用现有 `.venv`（OpenCV 4.10.0、NumPy 2.1.2、Torch 2.15.0+cpu）独立重建，15 项视图指纹与上述本机结果全部一致，对 B 仍为 2/15。见 [第二套现有环境复核](local_secondary_environment.json)。这次没有安装或更改依赖，说明切到本机已有的另一套 Torch/NumPy 环境仍不能复现 B 输入，尚不能据此确定根因。

C 本机研究数据指纹为 `e494bcb9…eaf`，B 为 `4f3a1790…af9`。该指纹同时覆盖来源文件、划分及所有训练/验证生成像素；不一致不能独立证明所有训练像素均已不同，训练生成像素的可比性需要 B 的完整数据清单继续核对。

## 之前的包与结果找到了什么

已检查 OneDrive（含桌面及微信缓存）、Downloads，以及 Documents、source 和本地 Desktop 下相关源码/产物名；排除了依赖、游戏和原始图片目录。对相关 ZIP 只读取目录，不执行其中代码。

| 找到的内容 | 判定 |
| --- | --- |
| `DS4023_TrafficSign_ResNet18_Spark_20261117.zip` | 已找到此前的正式训练输入包；51,919 项，包含代码、数据和官方初始化，0 个 `model.pt`、0 个 `completion.json` |
| `DS4023_TrafficSign_Spark_20261117.zip` | 旧 SmallCNN 的训练输入包，不是正式 ResNet18 结果包 |
| `DS4023_TrafficSign_Transfer_20261112.zip` | 旧 Windows 迁移包，同样不含训练模型或 A 评估源码 |
| `results/study_v1/` | 历史 SmallCNN 的 15 次完成运行与报告 |
| `results/resnet_engineering_smoke/` | ResNet18 六种模式的小规模工程检查，不是正式 15 次 |
| `MS2_ResNet_Code_and_Results.zip` | 另一个 ResNet20/56 论文实验项目，不属于本交通牌 C 实验 |

未找到 `a_conditions.py`、`evaluate_a13.py`、`A_LINUX.sh`、正式 `study_resnet18_spark_v1` 或 Spark 运行环境记录。此结论仅覆盖已搜索的本机路径及相关包；没有连接实体 Spark，不能由此断言远端训练从未执行。

## 下一步如何解除阻塞

1. **先在 Spark 查正式结果。**在 Spark 的 Linux 终端可运行 `find "$HOME" -type f -name spark_runtime.json -print`。若找到，先确认同目录是否含 15 次 ResNet18 模型及完成标记，再使用现有 `verify` 验收；先确认产物位置再决定是否重训。参考 [Spark 完整操作步骤](../../ResNet18_Spark_完整操作步骤.md)。
2. **补 A 的原实现或冻结输入。**需要 `a_conditions.py`、`evaluate_a13.py`、对应 `conditions.json`、`images.csv` 和原始数据审计记录。附件没有规定 SHA 派生随机种子的精确编码，自行仿写不能保证像素一致。
3. **补 B 的正式产物。**至少提供 12 次模型与冻结配置、逐轮训练日志、逐图预测、特征、研究数据清单和完成指纹，才能复算指标、重载并核对训练像素。
4. **由 A 记录统一输入口径，B/C 复核。**同一十三条件评估集和对应视图指纹对齐后，才重新评估并作配对。单纯重新评估不能自动解决未验明的训练生成像素差异。
5. **额外推理成本单独验收。**B 的 Mac CPU 秒数不与 Spark GPU 秒数求速度比；同机、同 batch、同预热和重复次数下测量，视图预处理与网络评估分开记录。原成本表仍作为各自环境的记录保留。

当前五份冻结训练源码保持不变。C 代码准备状态没有退回，但“15 次正式实验、十三条件跨成员比较、额外成本结论”仍未验收。

## 复跑此次交接审计

取得完整 B 交接快照并放到 `handoffs/B_20261122/` 后，在项目根目录使用 Python 3，指定一个尚不存在的输出目录。

```bash
python3 tools/audit_b_handoff.py --source handoffs/B_20261122 --output results/b_handoff_review_20261122/recheck_v2
```

本机 Windows 可使用现有项目 Python 执行同一脚本。该工具只需标准库，不训练、不加载权重；结构审计通过不等于完整训练或跨成员可比性验收通过。

最终回归运行 67 项，66 项通过、1 项因 Windows 符号链接权限跳过，0 失败、0 错误。记录见 [测试摘要](regression_tests.json)。这些测试包含小样本训练与模拟运行，不构成正式 ResNet18 15 次训练结果。
