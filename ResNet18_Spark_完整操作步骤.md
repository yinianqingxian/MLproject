# DGX Spark 上的 ResNet18 训练与报告操作步骤

适用对象：在实体 NVIDIA DGX Spark 上运行 C 部分实验的同学。本文对应 `DS4023_TrafficSign_ResNet18_Spark_20261117.zip`，从解压到完成 15 次训练、核验和生成中文报告。按顺序执行，每一步成功后再继续；出现错误时先停止后续步骤。

操作步骤已按交付包中的脚本核对。实体 Spark 的镜像构建、GPU 训练和耗时需要在目标机器上验证；当前包内没有正式 ResNet18 实验成绩。训练负责人执行下列步骤，Docker 权限或 DGX OS 问题交给机器管理员；代码错误保留日志后交给项目协作者排查。

## 1. 把两个交付文件复制到 Spark

在 Spark 终端创建接收目录：

```bash
mkdir -p "$HOME/ds4023-incoming"
```

通过 U 盘、局域网或已有的文件传输方式，把以下两个文件放进 Spark 用户主目录下的 `ds4023-incoming` 文件夹，文件名保持原样：

- `DS4023_TrafficSign_ResNet18_Spark_20261117.zip`
- `DS4023_TrafficSign_ResNet18_Spark_20261117.zip.sha256`

使用文件名中带 **ResNet18** 的包。包里已有训练代码、完整 GTSRB 原始数据、固定划分、官方 ImageNet 初始化权重和 Spark 配置，不需要复制 Windows 虚拟环境或另行下载权重。首次构建容器仍需联网获取 NVIDIA 基础镜像及缺失依赖。

## 2. 确认 Spark 环境并校验压缩包

在 Spark 终端执行：

```bash
uname -m
nvidia-smi
docker ps
python3 --version
df -h "$HOME"
```

架构应为 `aarch64`，`nvidia-smi` 应显示 GB10 GPU，当前用户应能运行 Docker，且主目录所在磁盘有足够空间存放镜像、模型和导出特征。镜像和训练结果会继续占用空间，不能仅按约 401 MiB 的压缩包大小预留空间。

若 Docker 无权限或无法连接服务，先由管理员修复。下面的脚本不要求安装宿主机 Python 依赖，也不替换驱动或 CUDA。不要运行原 Windows 安装器或通用 x86 Linux 安装器。

进入接收目录并验证下载/复制是否完整：

```bash
cd "$HOME/ds4023-incoming"
sha256sum -c DS4023_TrafficSign_ResNet18_Spark_20261117.zip.sha256
```

必须看到：

```text
DS4023_TrafficSign_ResNet18_Spark_20261117.zip: OK
```

交付 ZIP 的 SHA256 为：

```text
8f25a35b997f65e4778e7a64e9fdc8fdd5a2b4093ff99f9d20bd265982c4d3f3
```

校验失败时重新复制文件，不继续解压或训练。

## 3. 解压到新目录并核对包内文件

以下命令创建新工作目录。如果目录已经存在，`mkdir` 会失败并阻止这一条解压命令；请保留已有目录，先核对它是否为之前的训练工作目录。

```bash
mkdir "$HOME/ds4023-resnet18-20261117" &&
python3 -m zipfile -e DS4023_TrafficSign_ResNet18_Spark_20261117.zip "$HOME/ds4023-resnet18-20261117"
```

进入解压后的项目根目录。后面的所有命令都在这里执行：

```bash
cd "$HOME/ds4023-resnet18-20261117/DS4023_TrafficSign"
python3 tools/verify_spark_package.py --progress
```

末尾 JSON 应含 `"status": "passed"`，原始数据应为 39,209 张训练图和 12,630 张测试图。包内还会核验官方权重和冻结划分。此步骤只校验交付内容，不需要 GPU，也不安装包。

固定项目根目录为：

```text
$HOME/ds4023-resnet18-20261117/DS4023_TrafficSign
```

## 4. 构建容器、运行测试和单轮试跑

先创建日志目录。只创建 `logs`，不要预先创建正式实验输出目录。

```bash
mkdir -p logs
bash SPARK.sh --help
```

构建并检查环境：

```bash
bash -o pipefail -c 'bash SPARK.sh setup 2>&1 | tee logs/setup.log'
```

首次执行会构建 `ds4023-c-resnet18-spark:25.11`，随后进行实际 GPU 前向、反向和优化器更新检查。成功状态应含：

```text
spark_environment_and_data_ready
```

环境证据保存在 `results/spark_environments/`。`setup` 已包含检查，不必紧接着重复运行 `check`；以后需要单独复查时可运行 `bash SPARK.sh check`。

运行包内测试：

```bash
bash -o pipefail -c 'bash SPARK.sh tests 2>&1 | tee logs/tests.log'
```

末尾应为 `OK`，不应有 `FAILED` 或错误。然后只运行一轮 mixed/dual/seed42，确认整条训练和导出流程可执行：

```bash
bash -o pipefail -c 'bash SPARK.sh smoke 2>&1 | tee logs/smoke.log'
```

成功时会打印以这段文字开头的消息：

```text
Completed and verified 1 official ResNet18 runs in
```

单轮结果位于 `results/resnet18_spark_smoke/`，不计入正式 15 次实验。以上命令的外层启用了 `pipefail`，即使 `tee` 成功写入日志，也不会掩盖训练程序的失败。

## 5. 后台运行正式 15 次实验并自动生成报告

默认矩阵如下，每一行运行三个种子，共 15 次；无需手动启动 15 个命令。

| 训练配置 | 输入模式 | 随机种子 | 次数 |
| --- | --- | --- | --- |
| `repeat_clean` | `dual` | 42、43、44 | 3 |
| `targeted_motion` | `dual` | 42、43、44 | 3 |
| `mixed` | `dual` | 42、43、44 | 3 |
| `mixed` | `fixed` | 42、43、44 | 3 |
| `mixed` | `dual_denoised` | 42、43、44 | 3 |

骨干使用 Torchvision 官方 ImageNet 预训练 ResNet18，保留课程要求的双视图门控融合和固定平均对照。正式配置固定为每次 10 轮、batch 128、AdamW 学习率 0.003、CUDA、6 个 CPU 线程。不要为占满内存而改变 batch、种子或轮数。

下面整段复制执行一次。它会依次执行 `train → verify → report`，前一步失败就停止，并把退出状态写入独立日志目录。`nohup` 让任务在 SSH 断开后继续运行；机器仍需保持开机且不能休眠。

```bash
bash -euo pipefail <<'BASH'
study_dir=results/study_resnet18_spark_v1
report_dir=report/c_part_resnet18_spark
task_log_dir=$(mktemp -d logs/formal_XXXXXXXX)
printf '%s\n' "$task_log_dir" > logs/latest_formal_log_dir.txt

nohup bash -c '
  set -uo pipefail
  log_dir="$1"
  study_dir="$2"
  report_dir="$3"
  if bash SPARK.sh train --output-dir "$study_dir" &&
     bash SPARK.sh verify --output-dir "$study_dir" &&
     bash SPARK.sh report --study-dir "$study_dir" --report-dir "$report_dir"; then
    printf "0\n" > "$log_dir/exit_code.txt"
    printf "All 15 runs verified and report generated.\n" > "$log_dir/SUCCESS.txt"
  else
    result=$?
    printf "%s\n" "$result" > "$log_dir/exit_code.txt"
    exit "$result"
  fi
' _ "$task_log_dir" "$study_dir" "$report_dir" </dev/null > "$task_log_dir/pipeline.log" 2>&1 &

printf '%s\n' "$!" > "$task_log_dir/pipeline.pid"
printf '日志目录：%s\n' "$task_log_dir"
BASH
```

默认正式结果目录必须尚不存在。不要重复粘贴启动块，也不要同时启动第二个训练、测试、试跑、检查或报告进程。容器退出后，挂载在项目目录里的结果会保留在 Spark 磁盘上。

如果全程在 Spark 本机终端操作，也可以保持终端打开，依次运行下面三个命令；这是上一段后台流程的替代方式，两种方式选一种：

```bash
bash SPARK.sh train
bash SPARK.sh verify
bash SPARK.sh report
```

后一条命令只能在前一条成功后执行。关闭普通前台训练终端或按 Ctrl+C 可能中断实验，因此远程操作优先使用前面的后台流程。

## 6. 查看进度并确认全部完成

重新连接 SSH 后，先回到项目目录并打开日志：

```bash
cd "$HOME/ds4023-resnet18-20261117/DS4023_TrafficSign"
task_log_dir=$(cat logs/latest_formal_log_dir.txt)
tail -n 80 -f "$task_log_dir/pipeline.log"
```

这里按 Ctrl+C 只停止查看日志，不会终止后台任务。可在另一个终端偶尔运行 `nvidia-smi` 查看 GPU 状态。

日志会显示每个实验的 `epoch 1/10` 到 `epoch 10/10`。第 10 轮结束后还需要评估、导出模型和特征并计算指纹，因此看到 `epoch 10/10` 不代表该实验已经完整结束。

后台流程全部成功时，日志中应依次出现：

```text
Completed and verified 15 official ResNet18 runs in
Verified 15 completed official ResNet18 runs
Validated 15 runs, 75 metric rows, 63000 validation predictions; report:
```

查看后台退出状态和成功标记：

```bash
cat "$task_log_dir/exit_code.txt"
cat "$task_log_dir/SUCCESS.txt"
```

应分别看到 `0` 和 `All 15 runs verified and report generated.`。如果 `exit_code.txt` 尚不存在，任务可能仍在运行，也可能异常终止；结合日志及下列进程信息判断：

```bash
ps -p "$(cat "$task_log_dir/pipeline.pid")" -o pid,etime,args
```

退出码非 0 或日志出现异常时，按第 8 步处理。只统计模型文件或目录数量不能替代 `verify` 的完整验收。

## 7. 打开报告并检查交付结果

默认输出位置如下，全部相对于项目根目录：

| 内容 | 路径 |
| --- | --- |
| 中文完整报告 | `report/c_part_resnet18_spark/C_REPORT.md` |
| 报告索引 | `report/c_part_resnet18_spark/README.md` |
| 汇总统计表 | `report/c_part_resnet18_spark/tables/` |
| 五类图表的 PNG 和 SVG | `report/c_part_resnet18_spark/figures/` |
| 报告来源指纹与数量清单 | `report/c_part_resnet18_spark/summary_manifest.json` |
| 15 次模型、逐轮日志、逐图预测、特征与完成标记 | `results/study_resnet18_spark_v1/runs/` |
| 冻结运行环境 | `results/study_resnet18_spark_v1/spark_runtime.json` |
| 镜像和 GPU 环境检查记录 | `results/spark_environments/` |
| 启动、试跑和正式流水线日志 | `logs/` |

用支持 Markdown 的编辑器打开 `C_REPORT.md`。报告包含各验证条件下的指标均值与标准差、按种子的配对比较、门控分布、干预实验、训练曲线和成本分析。五类图表为 `scores_with_sd`、`paired_changes`、`gate_distribution`、`training_curves`、`intervention_effects`。

可用宿主机 Python 读取报告清单，检查数量：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("report/c_part_resnet18_spark/summary_manifest.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "runs": 15,
    "metric_rows": 75,
    "validation_prediction_rows": 63000,
    "intervention_metric_rows": 180,
    "intervention_prediction_rows": 151200,
    "feature_archives": 90,
}
for name, value in expected.items():
    actual = manifest[name]
    if actual != value:
        raise SystemExit(f"FAIL: {name}={actual}, expected {value}")
    print(f"{name}: {actual}")
print("Report counts passed.")
PY
```

这只是数量复查；完整配置、数据、模型和预测一致性已由前一步 `verify` 和报告程序验收。报告属于固定开发验证划分，官方 GTSRB 测试集没有用于这 15 次实验的调参或测试集成绩汇报。

如只需带走可阅读的报告和图表，可额外生成报告 ZIP：

```bash
python3 -m zipfile -c ResNet18_Spark_report.zip report/c_part_resnet18_spark
```

该 ZIP 只包含报告、表格、图和来源清单。需要保留可复核的完整实验时，另行备份整个项目目录，包括 `results`、`logs` 和官方初始化权重。

## 8. 失败、重跑和恢复

**构建或环境检查失败：**查看 `logs/setup.log`，修复后重新运行 `setup`。若拉取 NVIDIA 镜像明确提示认证失败，在 Spark 本机运行 `docker login nvcr.io` 并按账户要求登录。不要提供凭据给协作者，也不要通过全量升级 PyTorch/Torchvision 绕过依赖约束。

**单轮试跑失败并留下半成品：**保留原目录，在修复原因后换一个新目录试跑：

```bash
bash -o pipefail -c 'bash SPARK.sh smoke --output-dir results/resnet18_spark_smoke_v2 2>&1 | tee logs/smoke_v2.log'
```

**正式训练中断或失败：**先确认旧任务已停止，保留失败目录和日志。当前程序不能从某个 epoch 的中间继续，已有不完整实验目录会使 `--resume` 拒绝运行。修复原因后，重新执行第 5 步的后台启动块，只把研究目录和报告目录两行改为：

```bash
study_dir=results/study_resnet18_spark_v2
report_dir=report/c_part_resnet18_spark_v2
```

这会在全新目录重新运行完整 15 次，不复用失败目录。后续检查和数量复查也要把路径改为 v2；不要把两套研究目录的实验拼在一起。

**只有完整实验和尚未创建的实验目录：**在镜像 ID、驱动与依赖、代码、数据、划分和配置完全一致时，`--resume` 可以核验并跳过已完成实验，再启动缺失实验。例如在保持连接的终端中执行：

```bash
bash SPARK.sh train --resume
```

若目录里只有初始化时写入的有效 `spark_runtime.json`，修复原始错误后，同一运行环境下也可用上述命令继续初始化。任何已存在的不完整实验都会阻止此方式。不要手改 `completion.json`、`status.json`、`spark_runtime.json` 或删除局部产物来绕过验收。

**15 次训练已完成，只是报告失败：**保留训练结果，排查报告日志后单独重跑：

```bash
bash SPARK.sh verify
bash SPARK.sh report
```

这两步不会重训模型。若使用 v2，应明确指定对应路径：

```bash
bash SPARK.sh verify --output-dir results/study_resnet18_spark_v2
bash SPARK.sh report --study-dir results/study_resnet18_spark_v2 --report-dir report/c_part_resnet18_spark_v2
```

手动补做报告成功后，以这次命令的退出状态和报告验收消息为准；先前后台流水线的失败状态文件不会自动改写。排错时保存失败命令、对应日志和 `results/spark_environments/` 环境报告，并记录使用的研究目录。
