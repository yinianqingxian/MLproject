# 官方 ResNet18 双视图融合 · DGX Spark 训练包

本包包含 Spark 训练必需的源码、测试、完整 GTSRB 数据、固定划分、官方 ResNet18 ImageNet 初始化权重和使用说明。按用户确认范围保留原双视图融合，仅将骨干换成官方 ResNet18。未包含 Windows 虚拟环境、安装器、wheel、缓存或历史训练结果。Spark 的 Docker 镜像与缺失依赖需要联网获取，本包不是离线容器镜像。

## 解压与校验

将 ZIP 和同名 `.zip.sha256` 文件复制到 Spark，先核对传输文件：

```bash
sha256sum -c DS4023_TrafficSign_ResNet18_Spark_20261117.zip.sha256
python3 -m zipfile -e DS4023_TrafficSign_ResNet18_Spark_20261117.zip SparkResNet18_20261117
cd SparkResNet18_20261117/DS4023_TrafficSign
python3 tools/verify_spark_package.py --progress
```

请使用新的解压目录，不覆盖已有研究。校验工具只依赖 Python 标准库，无需先安装 PyTorch；成功会显示 `status: passed`。包内 `PACKAGE_MANIFEST.json` 逐项记录文件大小和 SHA256。官方训练图应有 39,209 张、测试图 12,630 张，冻结划分共 3,780 行。

## 在 Spark 运行

在项目根目录依次执行：

```bash
bash SPARK.sh setup
bash SPARK.sh tests
bash SPARK.sh smoke
bash SPARK.sh train
bash SPARK.sh verify
bash SPARK.sh report
```

`setup` 使用 NVIDIA ARM64 PyTorch/Torchvision 容器，检查 GB10/CUDA、数据、官方初始化、绘图与六种 ResNet 输入模式的实际 GPU 优化步骤。`smoke` 是一轮工程试跑；`train` 才执行 15 次正式实验。完整操作、依赖保护和排错见 [DGX_SPARK.md](DGX_SPARK.md)，训练与评价协议见 [PROTOCOL.md](PROTOCOL.md)。

## 内容和输出

| 位置 | 内容 |
| --- | --- |
| `SPARK.sh`、`spark/` | Spark 容器构建、GPU 运行和缺包配置 |
| `experiments/` | 公共模型、固定实验矩阵、指标汇总及验收测试 |
| `tools/` | Spark 环境检查、恢复保护、数据检查辅助和本包校验 |
| `data/GTSRB/raw/` | 完整 43 类原始图像、标注和数据说明 |
| `results/pilot_split.csv` | 原样保留的七类轨迹划分，路径相对于 `data/GTSRB/raw/` |
| `assets/weights/` | 官方 ResNet18 ImageNet 初始化及 SHA256 来源说明 |
| `results/study_resnet18_spark_v1/` | 正式训练后新建，保存模型、配置、日志、预测、512 维特征和恢复环境 |
| `report/c_part_resnet18_spark/` | 汇总后新建，保存 ResNet 报告、统计表与图 |

本包不含本项目已经训练的 ResNet 结果；结果和报告目录在运行后创建。历史 SmallCNN CPU 实验已完成，仍保留在原工作目录。新 ResNet 方案的结构、短程训练和导出可在本机核验，实体 Spark 构建和正式 GPU 训练仍须在目标机完成，不能将包校验通过写成新 C 实验已完成。

## 失败与恢复

完整运行重新启动时可使用 `bash SPARK.sh train --resume`，先核对镜像、驱动、依赖、数据、配置及产物。半成品保留排错，换新的 `--output-dir` 重跑；不要手改完成标记。Docker 访问、网络或 GPU 检查失败时，按 [Spark 说明](DGX_SPARK.md) 处理，原始数据与冻结划分保持不变。
