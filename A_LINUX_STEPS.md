# A 在 Linux GPU 服务器上的操作步骤

> 目标：在**干净的 x86_64 Linux 机器**上跑完 A 的九次正式训练、十三条件评估和汇总。全部命令都在仓库根目录执行。
> **DGX Spark（ARM64 / GB10）不要用本文**：pip 的 CUDA wheel 在 aarch64 上不通用，Spark 请按 [A_SPARK_STEPS.md](A_SPARK_STEPS.md) 用 NVIDIA ARM64 容器运行 `bash A_SPARK.sh …`。
> 协议与冻结参数见 [A_PROTOCOL.md](A_PROTOCOL.md)。本文只讲操作、预期产物和排错。

## 0. 前提

- x86_64 Linux，已安装 NVIDIA 驱动（`nvidia-smi` 能跑）。
- Python ≥ 3.10（`python3 -V`）。
- 能联网装 PyTorch（或已有可用的 CUDA 版 Torch/Torchvision 环境）。
- 磁盘 ≥ 10 GB（数据 465 MB，九个运行约 3–5 GB，评估产物再约 2 GB）。

时间预算（单卡，按 RTX 4090 级估算，实际以 `check` 的探针输出为准）：九次训练合计约 20–40 分钟，十三条件评估约 5–10 分钟，其余步骤都是分钟级。

## 1. 把代码放到服务器

任选一种。

**A. 从 GitHub（推荐，先确认 A 的分支已 push）**

```bash
git clone -b a/data-protocol-evaluation https://github.com/yinianqingxian/MLproject.git
cd MLproject
```

**B. 从这台 Windows 直接传**

```powershell
# 在 Windows PowerShell 里执行
scp -r "C:\Users\Weisifengbuxi\Desktop\ML project\MLproject" user@server:/path/to/
```

不要传 `.venv-a-check`、`PACKAGE_MANIFEST*.json` 这类本机文件；`.git` 可以传，也可以不要。

## 2. 放数据与官方权重

**A. 从这台 Windows 传（推荐，数据已在本地，465 MB）**

```powershell
# Windows：打包 GTSRB_dataset 的内容（它等价于 data/GTSRB/raw）
tar -czf a_gtsrb_raw.tgz -C "C:\Users\Weisifengbuxi\Desktop\ML project\GTSRB_dataset" .
scp a_gtsrb_raw.tgz user@server:/path/to/MLproject/
```

```bash
# Linux：解到 data/GTSRB/raw
cd /path/to/MLproject
mkdir -p data/GTSRB/raw
tar -xzf a_gtsrb_raw.tgz -C data/GTSRB/raw
```

**B. 在服务器直接从官方归档下载**

```bash
cd data/GTSRB/raw
BASE=https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
# 按页面上的三个 ZIP 下载后解压，整理成：
#   GTSRB/Final_Training/Images/<类别>/*.ppm
#   GTSRB/Final_Test/Images/*.ppm
#   GT-final_test.csv
```

官方权重（`assets/weights/` 不在仓库里，必须自己下）：

```bash
curl -fL https://download.pytorch.org/models/resnet18-f37072fd.pth \
  -o assets/weights/resnet18-f37072fd.pth
printf '%s\n' 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec  assets/weights/resnet18-f37072fd.pth' | sha256sum -c -
```

## 3. 核对数据规模

```bash
find data/GTSRB/raw/GTSRB/Final_Training/Images -name '*.ppm' | wc -l   # 应为 39209
find data/GTSRB/raw/GTSRB/Final_Test/Images -name '*.ppm' | wc -l        # 应为 12630
wc -l results/pilot_split.csv                                            # 应为 3781（含表头）
sha256sum data/GTSRB/raw/GT-final_test.csv
```

## 4. 装环境

```bash
bash A_LINUX.sh setup
source .venv-a-linux/bin/activate      # 之后每个新终端都要执行
```

脚本会从 `nvidia-smi` 推断驱动支持的 CUDA 版本，选对应的 PyTorch wheel，再装 `requirements-a.txt`（固定 NumPy/OpenCV 版本，尽量与 Windows 的像素生成版本一致）。若固定版本没有 wheel，会退回兼容范围并打印警告。

已有厂商容器或团队统一环境时，跳过 `setup`，直接激活该环境并把 `A_PYTHON` 指向它的解释器。

## 5. 检查（必做）

```bash
bash A_LINUX.sh check
```

这一步会：

1. 用**正式 batch 128** 做一次真实的前向/反向/优化器更新，报告峰值显存和单步耗时，并估算每次训练分钟数。显存不够会直接报错，不会让你跑到一半才 OOM。
2. 重放冻结划分与增强预算，写出 `results/a_linux_environment/data_audit/`，并断言轨迹无交叉、预算正确。
3. 跑三组回归测试：数据协议、十三条件、汇总程序。

要让 `check` 通过，不允许为了让显存够而下调 batch 或开启混合精度；那会改变全组统一的训练协议。

## 6. 工程试跑

```bash
bash A_LINUX.sh smoke
```

`mixed × 种子 42 × 1 轮`，写入 `results/a_linux_smoke_v1/`。它**不计入**正式结果。通过即可进入正式训练。

## 7. 正式九次训练

```bash
tmux new -s a-train
bash A_LINUX.sh train 2>&1 | tee -a logs/a_train.log
```

- 矩阵：`raw` 输入 × `repeat_clean / targeted_motion / mixed` × 种子 `42/43/44` = 9 次。
- 每次 10 轮、350 次更新，输出到 `results/a_resnet18_linux_v1/`。
- 断开 SSH 会中断，所以务必在 tmux 里跑。
- 中断后：`bash A_LINUX.sh resume`。它只跳过已完整通过核验的运行，不支持从半个 epoch 续训；遇到半成品会拒绝，此时换新目录重跑（`A_OUTPUT=results/a_resnet18_linux_v2 bash A_LINUX.sh train`）。

## 8. 核验

```bash
bash A_LINUX.sh verify
```

只读核验九个运行：配置、六份评估产物、逐图预测、特征与文件指纹。正常输出 `Verified 9 completed official ResNet18 runs`。

预期产物：9 个运行目录、45 条常规指标、37,800 条验证预测、54 份 512 维特征。

## 9. 开发集十三条件评估

```bash
bash A_LINUX.sh eval13
```

加载九个已训练模型，在 840 张开发验证图上评估 13 个条件，写入 `results/a_eval13_linux_v1/`。不需要重新训练。

预期：117 条指标、98,280 条预测、117 份特征、117 行成本记录。

## 10. 汇总表格与图

```bash
bash A_LINUX.sh report
```

写入 `report/a_part/`：`tables/dev13_*.csv`、`figures/dev13_*.png`、`dev13_summary.md`。这张汇总表就是交给 D 的原始材料。

## 11. 带回什么

```bash
tar -czf a_linux_results.tgz \
  results/a_resnet18_linux_v1 \
  results/a_eval13_linux_v1 \
  results/a_linux_environment \
  results/a_linux_smoke_v1 \
  report/a_part \
  logs
```

至少要带回：`run_config.json`、每个运行的 `config.json`/`model.pt`/`metrics.csv`/`predictions.csv`/`features/`、`completion.json`、评估目录的 `conditions.json`/`completion.json`、`results/a_linux_environment/environment.json` 与 `pip-freeze.txt`。**不要只带最终准确率或截图。**

模型权重（`*.pt`）在 `.gitignore` 里，不要提交进仓库，用网盘或 `scp` 单独传。

## 12. 官方测试（全组方案冻结之后）

以下默认命令只评估 A 的研究目录。

```bash
bash A_LINUX.sh freeze     # 写出冻结清单，此时不要看测试分数
bash A_LINUX.sh test7      # 打开测试分数
```

全组正式测试使用单独的清单。将示例中的 B、C 路径替换为各自已验收的研究目录。

```bash
python experiments/evaluate_official_test.py --init-freeze results/a_test7_team_freeze.json \
  --study results/a_resnet18_linux_v1 --study results/b_study_xxx --study results/c_study_xxx
python experiments/evaluate_official_test.py --freeze results/a_test7_team_freeze.json \
  --output results/a_test7_team_v1 --device cuda
```

冻结后再改配置或权重，评估会拒绝运行。

## 13. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available() is False` | 装成了 CPU 版 wheel | 重装匹配驱动 CUDA 的 wheel；不要继续 |
| `Batch 128 does not fit` | 显存不足 | 换更大显存的卡，或全组统一改协议后重跑；不要自行降 batch 或开 AMP |
| `Run was trained on different generated pixels` | 训练与评估不在同一依赖环境 | 在同一环境重跑评估；或按协议记录像素差异 |
| `Output already exists` | 目录已存在，程序拒绝覆盖 | 换新目录名（`A_OUTPUT=...`） |
| `Study code fingerprints differ` | 训练后改了源码 | 换新输出目录重跑；不要手改指纹 |
| `Verified ... ` 报缺 artifact | 运行未完成或被改动 | 保留现场排错，换新目录重跑 |
| 固定版本装不上 | Python 版本太新，没有对应 wheel | `setup` 会自动退回兼容范围并警告；跨机器像素一致性需另核 |
