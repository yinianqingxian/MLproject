# 在 DGX Spark 上训练 C 部分

本页供在 NVIDIA DGX Spark 运行官方 ResNet18 双视图融合实验的训练者使用。用户已确认保留既有融合，仅替换共享骨干。已提供 ARM64 容器配置和统一的 `SPARK.sh` 入口。将项目复制到 Spark，在项目根目录依次执行：

```bash
bash SPARK.sh setup
bash SPARK.sh smoke
bash SPARK.sh train
bash SPARK.sh verify
bash SPARK.sh report
```

配置和本机可执行的命令测试已核对；**本交付环境为 Windows CPU；新 ResNet 只能在本机做工程核验，未在实体 Spark 上构建镜像或运行 GPU 训练**。目标机器只有在 `setup/check`、单轮 `smoke` 和完整训练均成功后，才能声称 Spark 运行通过。新 ResNet 与历史 SmallCNN 结果必须分开记录。原有 15 次分数不能作为 ResNet 成绩，官方预训练初始化不等于本项目已经训练完成。

## 为什么使用专用入口

Spark 使用 ARM64 CPU 和 GB10 GPU。原有 `setup_linux.sh`、`.venv-linux` 及 `requirements-linux*.txt` 面向 x86_64，不用于 Spark。这里固定采用 NVIDIA Spark PyTorch 示例列出的 `nvcr.io/nvidia/pytorch:25.11-py3`，其环境为 Python 3.12、CUDA 13.0.2 与 NVIDIA PyTorch 2.10 开发构建。该选择是固定的官方示例版本，不声称它是当前最新镜像。[NVIDIA Spark 示例](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-pytorch-fine-tune/README.md#supported-hardware-platforms)、[25.11 发行说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html)

容器复用并保护 NVIDIA 成套的 PyTorch、Torchvision、NumPy、SciPy 和依赖约束，在独立、可读取系统包的项目虚拟环境中补装缺失的 OpenCV、scikit-learn 或 Matplotlib。现有包健康时直接复用，不强制改成 Windows 锁定版本；缺包安装后再次核验核心版本与模块路径。最终使用的包以构建记录和运行环境报告为准。不要在此容器安装整个 Windows 或通用 x86 Linux 依赖文件。

## 1. 搬运和前提

先按 [仓库说明](README.md) 准备完整 GTSRB、固定划分和官方初始化权重。使用独立训练 ZIP 时，按 [解压步骤](ResNet18_Spark_完整操作步骤.md) 进入解压后的项目根目录。`spark/` 是很小的镜像构建上下文，原始数据和已有模型通过运行时挂载，不会复制进镜像。

在 Spark 的终端或 SSH 会话中进入项目目录，先运行：

```bash
uname -m
nvidia-smi
docker ps
df -h .
```

机器架构应为 `aarch64`，GPU 应为 GB10，Docker 当前用户应能连接服务，并为镜像下载、环境和结果保留空间。DGX OS 已包含 NVIDIA Container Toolkit/Docker GPU 集成；若 `docker ps` 权限不足，由机器管理员配置当前用户的 Docker 访问，脚本不修改用户组或驱动。[NVIDIA Docker 运行时说明](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)

GPU 驱动以 Spark 的 DGX OS 更新渠道维护。本配置不会替换系统驱动或安装宿主机 CUDA Toolkit；实际 GPU 核验比仅检查版本号更直接。若 NGC 拉取提示需要认证，在 Spark 本机运行 `docker login nvcr.io`，按 NGC 指引使用账户凭据，凭据不用提供给协作者。

## 2. 构建和检查

```bash
bash SPARK.sh --help
bash SPARK.sh setup
bash SPARK.sh check
```

`setup` 在 Spark 构建 `ds4023-c-resnet18-spark:25.11` 并检查运行环境；后续检查会复用镜像。成功状态为 `spark_environment_and_data_ready`，包括：

- Linux ARM64、Python 3.12、可用 CUDA，以及 GB10 的计算能力 12.1。
- 39,209 张训练图、12,630 张测试图的目录完整性，及与原协议一致的 3,780 行固定划分；不使用官方测试预测调参。
- 实际 PPM 解码、七类 Macro-F1 和无显示器的 PNG 绘图。
- 原图、CLAHE、去噪 CLAHE、门控、固定平均和去噪门控六种官方 ResNet18 输入模式的 FP32 GPU 前向、反向及 AdamW 一步更新；同时核对初始化权重完整 SHA256。

环境报告保存于 `results/spark_environments/<镜像ID>.json`，包含镜像标识、驱动、PyTorch/CUDA/cuDNN、科学计算包实际版本、模块路径和划分指纹。它是环境探针证据，不是正式实验已经完成的标志。报告不依赖离散显卡的专用显存读数，Spark 的统一内存不会被当作 CUDA 不可用。[GB10 计算能力](https://developer.nvidia.com/cuda/gpus)

启动时设置 `NVIDIA_IMEX_CHANNELS=0`，对应 NVIDIA 25.11 记录的 NVSHMEM 运行问题。维持 eager/FP32 程序，不启用 `torch.compile`、AMP、FP4 或分布式训练。[25.11 已知问题](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html#known-issues)

## 3. 单轮试跑和正式训练

```bash
bash SPARK.sh smoke
bash SPARK.sh train
```

`smoke` 只运行 mixed/dual/seed42 的一轮，写入 `results/resnet18_spark_smoke`。`train` 执行 C 的固定 15 次训练，默认写入 `results/study_resnet18_spark_v1`。缓存干扰视图仍为 32×32，每批再应用官方 Resize256/CenterCrop224/ImageNet 归一化，骨干实际输入 224×224。batch 128、AdamW 学习率 0.003、10 轮和种子 42/43/44 保持原协议；训练使用最后一轮，不选择最佳验证轮。不用将 batch 放大来占满 128GB 内存，也不预设 Spark 必然比当前 CPU 更快。

已有目录默认拒绝覆盖。完整运行可以在同样镜像、代码与参数下使用 `--resume` 校验后跳过；半成品不支持从中断轮次续训。需要重新训练时改用独立目录：

```bash
bash SPARK.sh smoke --output-dir results/resnet18_spark_smoke_v2
bash SPARK.sh train --resume
bash SPARK.sh train --output-dir results/study_resnet18_spark_v2
```

挂载目录中的结果会保留到 Spark 宿主机；容器结束不删除数据或模型。启动器使用当前用户的 UID/GID，正常运行生成的文件由当前用户拥有。更换镜像、依赖、训练参数或代码后使用新目录，不通过修改完成标记来绕过校验。

每个 Spark 研究目录另有 `spark_runtime.json`，在训练前冻结镜像 ID、驱动、核心/应用依赖及运行开关。恢复时先逐项核对该记录，再由公共入口核验数据、训练配置和产物指纹；缺少记录的其他平台目录不接受 Spark 恢复。若因参数错误或预检查失败而只留下这份环境记录，在修正参数且运行环境相同后可使用 `--resume` 继续初始化。`verify` 是只读产物核验，不要求当前环境等于训练环境。

## 4. 验收、报告和其他命令

```bash
bash SPARK.sh tests
bash SPARK.sh verify
bash SPARK.sh report
```

`verify` 检查 15 次运行的配置与产物指纹；`report` 再独立复算逐图指标、核对研究级冻结配置和特征对齐，输出 Spark 的统计表与图。默认对应如下：

| 内容 | 位置 |
| --- | --- |
| 镜像环境证据 | `results/spark_environments/` |
| 单轮工程核验 | `results/resnet18_spark_smoke/` |
| 正式模型、配置、训练日志、特征和预测 | `results/study_resnet18_spark_v1/runs/` |
| 研究级镜像/驱动/依赖恢复保护 | `results/study_resnet18_spark_v1/spark_runtime.json` |
| 汇总报告 | `report/c_part_resnet18_spark/C_REPORT.md` |
| 原 Windows 结果与报告 | `results/study_v1/`、`report/c_part/` |

若使用第二版结果目录，核验和报告也应指向它：

```bash
bash SPARK.sh verify --output-dir results/study_resnet18_spark_v2
bash SPARK.sh report --study-dir results/study_resnet18_spark_v2 --report-dir report/c_part_resnet18_spark_v2
```

Spark 与 Windows 的框架构建、硬件和浮点执行存在差异，不能要求权重 SHA256 或分数完全相同，也不能将两套结果混入同一个三种子均值。每份研究的配置、数据及代码指纹和逐图预测是各自的比较依据。官方测试继续保留到全组方案冻结后使用。

## 5. 失败处理

| 现象 | 处理 |
| --- | --- |
| 提示 x86_64 或错误架构 | 回到实体 Spark/DGX OS 上执行 `SPARK.sh`，使用 ARM64 镜像；不要用原通用 Linux 安装器。 |
| Docker 无权限或守护进程不可用 | 由管理员修复 Docker 访问与服务，再重跑 `setup`。 |
| NGC 拉取认证失败 | 在 Spark 本机认证 NGC，确认网络与镜像访问。 |
| 缺包安装与 NVIDIA 约束冲突 | 保留完整构建日志；不取消核心依赖约束或全量升级 PyTorch/NumPy，按日志定位缺包。 |
| CUDA/GB10 检查或实际 kernel 失败 | 确认使用官方 ARM64 NVIDIA 镜像和 GPU 透传，保存驱动与检查输出，按 DGX OS 文档维护驱动。 |
| 划分或指纹不一致 | 核对复制是否完整，保留原结果并恢复冻结划分，不重新抽样。 |
| 输出目录已存在或中断运行不完整 | 完整运行使用 `--resume`；半成品保留排错，换新目录重跑。 |

负责训练的成员保存失败命令、构建日志和环境报告用于协作排错。现有 Windows 研究可作为参考保留，不需要删除或重训它来修复 Spark 环境。
