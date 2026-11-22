# C 官方 ResNet18 骨干迁移

> 2026-11-17。状态：代码与本机工程核验通过，正式 15 次实验待 Spark 执行。用户已确认保留双视图融合，只将共享骨干替换为官方 ResNet；不另行设计残差块、stem、注意力或投影层。

## 固定决定与范围

使用 torchvision 官方 `resnet18` 和 `ResNet18_Weights.IMAGENET1K_V1`。保留其卷积、BatchNorm、残差层、下采样和平均池化；取官方平均池化后的 512 维表示，分类头适配七类。保留既有学习门控、固定平均及去噪视图消融，不将融合方法写成新发明。完整微调共享骨干与分类头。

依据：[原始 ResNet 论文](https://arxiv.org/abs/1512.03385)、[torchvision 官方 ResNet18 API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。工作定位为官方模型应用与既定融合对照，不宣称残差结构创新。

冻结七类与轨迹划分、2,940/840 图像、4,410 训练条目、三种增强、五种开发条件、种子 42/43/44、AdamW 0.003 和十轮最终轮评价。为保持干扰样本一致，仍先在既有 32×32 图上生成干扰与 CLAHE，再对每批视图应用官方权重的 Resize256/CenterCrop224/归一化；不改官方 stem，也不添加随机翻转或随机裁剪。新输入及 ImageNet 预训练是新实验条件，不能把旧分数当作 ResNet 结果。

## 接口与文件

| 文件 | 职责 |
| --- | --- |
| `experiments/resnet_robustness_study.py` | `ResNet18Fusion(fusion, pretrained=True, weights_path=None)`、`fit_run(run_dir, config, train_views, train_targets, evaluations)`；六种输入沿用既有名称，导出 512 维特征 |
| `experiments/run_resnet_study.py` | 复用 `run_c_study.prepare_data()` 和原矩阵；CLI 保持 epochs/threads/seeds/inputs/augmentations/resume/verify-only/device/output-dir，并增加 weights-path |
| `experiments/summarize_resnet_study.py` | 仅汇总完整的官方 ResNet18 配置、512 维特征和同一批 15 次运行；复算指标并输出新报告 |
| `assets/weights/resnet18-f37072fd.pth` | 官方 ImageNet 初始化，下载后保存完整 SHA256；模型训练和重载不隐式下载其他权重 |
| Spark 入口与包 | 默认转到独立 ResNet 研究目录，保护容器内已匹配的 torchvision；保留旧版本源码、模型和 ZIP |

配置额外记录 `backbone=torchvision.models.resnet18`、`feature_dim=512`、`network_input_size=224`、`weights_id=ResNet18_Weights.IMAGENET1K_V1`、`weights_sha256`、torchvision 版本及路径。`image_size=32` 表示缓存视图与干扰的尺寸，不是骨干输入尺寸。权重路径使用项目内相对路径；单分支与融合均经过同一官方预处理。

每运行仍保存最终模型、配置、日志、五条指标、4,200 条预测和六份特征；学习门控另存三种固定 g 干预。继续使用旧的 CSV/NPZ 字段，特征维度从 32 改为 512。完成标记与指纹校验不允许静默覆盖。新报告必须明确骨干、预训练、输入尺寸和结果尚未完成的状态。

## 实施和验收

- [x] 准备独立、匹配 Torch/Torchvision 的本机 CPU 核验环境及官方权重，不修改既有 `.venv` 或 NVIDIA 系统包。
- [x] 验证官方骨干结构、512 维输出、共享编码器、门控端点及梯度；实现训练和权重/特征导出。
- [x] 实现新运行入口和 ResNet 汇总入口，拒绝混入旧 SmallCNN 或不同权重的产物。
- [x] 更新 Spark 的 torchvision 保护、ResNet GPU 探针、恢复环境记录及独立输出路径。
- [x] 执行小样本真实 CPU 训练/重载，核验旧 15 次结果未变；独立代码审查通过。
- [x] 重新打包完整数据、官方权重、必需代码和新说明，ZIP 名含 ResNet18，不覆盖旧包。

新方案的 15 次正式训练由实体 Spark 执行。本机仅验证真实官方模型、短程训练和导出，不声称新的正式分数已经产生。历史 SmallCNN C 范围已完成；切换后的 C 只有待 Spark 正式训练并汇总后才能标记为实验完成。

A/B 后续用于直接比较的 raw/CLAHE 单分支，也应使用新的公共 ResNet 入口、相同官方初始化、输入变换及训练预算。旧 SmallCNN 单分支分数不与本次 ResNet 融合分数当作同骨干消融对照。

## 已取得的本机证据

官方初始化完整 SHA256 已核对。`.venv-resnet-test` 使用项目内虚拟环境，只读继承本机现有、已配对的 Torch 2.5.1/Torchvision 0.20.1，并在项目虚拟环境补装 OpenCV，不改变原 `.venv` 或全局依赖。这是工程核验环境，不冒充 Spark 的 NVIDIA 2.10 环境。

`results/resnet_engineering_smoke/checkpoint_verification.json` 记录六种模式各一次真实官方预训练短程训练：每类一张训练和一张验证图、单轮。全部 252 张特征/预测/门控导出和 210 条固定门控干预预测经权重重载核对通过，512 维特征及官方预处理成立。这些小样本数值不形成正式性能结论，也不随新训练包作为已完成模型成绩交付。

最终本机全套 54 项验收：53 项通过，1 项因 Windows 缺少符号链接创建权限而跳过。Bash 通过模拟 Docker 核验，未启动实际容器。独立审查发现的 Spark 首次预留目录与新运行入口不兼容、汇总缺少研究级数据清单/划分字节核验两项问题均已修复并复核；没有未解决的重要代码发现。旧 15 次运行及三份冻结训练源码已重新核对，保持不变。

新版包为 `../DS4023_TrafficSign_ResNet18_Spark_20261117.zip`，420,153,609 字节（约 401 MiB），包含 51,918 个内容文件及清单。全部原始数据源指纹、压缩包 SHA256/大小/CRC、官方初始化权重、新旧冻结训练源码、独立说明和 Python/Bash 文件均已核验。ZIP 外提供 `.sha256` 和 `.verification.json`，包内 `03` 是打包时的计划快照，最终交付核验以旁附 JSON 及本记录为准。
