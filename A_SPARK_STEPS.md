# A 在 DGX Spark 上的完整命令清单

> 目标机器：`jeremy@spark-d01d`（DGX Spark，ARM64 aarch64，GB10，CUDA 13.0，统一内存）。
> 你的操作路径：Windows 本机 → 远控 Mac → Mac 上 SSH 进 Spark。
> 本文按顺序给出每一步要敲的命令。原理见 [DGX_SPARK.md](DGX_SPARK.md)，协议见 [A_PROTOCOL.md](A_PROTOCOL.md)。
> 普通 x86_64 Linux 服务器请改用 [A_LINUX_STEPS.md](A_LINUX_STEPS.md)；**Spark 上不要用 `A_LINUX.sh setup`**（那是 x86 的 pip wheel 路线）。

## 目标文件夹

所有东西都放这一个目录，方便整体打包带回：

```text
~/ds4023_a/
├── MLproject/              # 仓库（代码、结果、报告都在这下面）
├── transfer/               # 上传的压缩包放这里
└── logs/                   # 训练日志
```

Spark 上的结果目录固定为（与 Windows 结果分开，不混进同一份均值）：

| 内容 | 路径 |
|---|---|
| 九次训练 | `results/a_spark_v1/` |
| 试跑 | `results/a_spark_smoke_v1/` |
| 十三条件评估 | `results/a_eval13_spark_v1/` |
| 环境与预算记录 | `results/a_spark_environment/` |
| 汇总表格与图 | `report/a_part_spark/` |

---

## 第 0 步（Windows）：把分支推到 GitHub

Spark 上要 `git clone`，所以分支得先上去。在 Windows PowerShell：

```powershell
cd "C:\Users\Weisifengbuxi\Desktop\ML project\MLproject"
git status
git log --oneline main..HEAD
git push -u origin a/data-protocol-evaluation
```

推不上去（没有凭据）就用打包的 fallback，跳过 GitHub：

```powershell
cd "C:\Users\Weisifengbuxi\Desktop\ML project\MLproject"
git bundle create "..\a_branch.bundle" main a/data-protocol-evaluation
```

## 第 1 步（Windows）：准备要上传的两个文件

数据包已经生成好了：

```text
C:\Users\Weisifengbuxi\Desktop\ML project\a_gtsrb_raw.tgz     340.5 MB
```

校验值（传完在 Spark 上对一下）：

| 文件 | sha256 |
|---|---|
| `a_gtsrb_raw.tgz` | `b46ebf9caf1d9542744151ede54d56190480a784cb97ecaafcd52b888b432b60` |
| 包内 `GT-final_test.csv` | `58a660fab5465cce66d7334eb9422ae09ed16186ad4500766b59b5a077a1c12f` |
| 仓库自带 `results/pilot_split.csv` | `de914a9199138474115a14776a61574651653ed87c72a26b9804bb35c5aa5359` |
| 官方 ResNet18 权重 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

包里是 `./GTSRB/…` 和 `./GT-final_test.csv` 共 51,839 张 PPM，解压到 `data/GTSRB/raw/` 正好是对的结构。

用你的远控软件把 `a_gtsrb_raw.tgz`（和 `a_branch.bundle`，如果你走了 fallback）拖到 Mac 上，比如 `~/Downloads/`。

## 第 2 步（Mac）：SSH 进 Spark

```bash
ssh jeremy@spark-d01d
```

## 第 3 步（Spark）：建文件夹并拿代码

```bash
mkdir -p ~/ds4023_a/transfer ~/ds4023_a/logs
cd ~/ds4023_a

# 情况 A：分支已推到 GitHub
git clone -b a/data-protocol-evaluation https://github.com/yinianqingxian/MLproject.git MLproject

# 情况 B：用 bundle（先把 a_branch.bundle 传到 ~/ds4023_a/transfer/）
# git clone -b a/data-protocol-evaluation ~/ds4023_a/transfer/a_branch.bundle MLproject

cd ~/ds4023_a/MLproject
git log --oneline -3
```

## 第 4 步（Spark）：先看清楚这台机器的状态

```bash
cd ~/ds4023_a/MLproject
bash A_SPARK.sh env
```

它会打印架构、GPU 当前占用、Docker 是否可用、磁盘/内存、以及数据、划分、权重是否就位。

**这一步要特别看两件事：**

1. `nvidia-smi` 里有一个 `VLLM::EngineCore` 进程占了约 88 GB、GPU 利用率 95%。它不是我起的，也不该我停。如果它在跑，训练能跑但会明显变慢；如果它继续涨内存，可能挤到我们。开训前先想好要不要让机主停掉它。
2. `Docker：FAIL` 说明当前用户连不上 Docker 守护进程，需要机主配置（见 DGX_SPARK.md），这一步不过后面都白搭。

## 第 5 步（Spark）：放数据

先看机器上是不是已经有现成的 GTSRB（这个项目之前给 Spark 准备过训练包，很可能已经传过）：

```bash
ls -d ~/DS4023_TrafficSign* ~/*Spark* ~/*/DS4023_TrafficSign* 2>/dev/null
find ~ -maxdepth 5 -type d -name 'Final_Training' 2>/dev/null
```

**如果有**，直接软链接过来，不用再传 467 MB：

```bash
cd ~/ds4023_a/MLproject
ls ~/DS4023_TrafficSign/data/GTSRB/raw/GTSRB/Final_Training/Images | head -3
ln -s ~/DS4023_TrafficSign/data/GTSRB/raw data/GTSRB/raw
ls data/GTSRB/raw/GT-final_test.csv
```

（路径按上面 `find` 的真实结果替换。`raw/` 里应该有 `GTSRB/` 和 `GT-final_test.csv`。）

**如果没有**，从 Mac 传（Mac 终端新开一个窗口，不要占用 Spark 的 SSH 会话）：

```bash
scp ~/Downloads/a_gtsrb_raw.tgz jeremy@spark-d01d:~/ds4023_a/transfer/
```

回到 Spark 解压：

```bash
cd ~/ds4023_a
tar -tzf transfer/a_gtsrb_raw.tgz | head -3        # 应看到 ./GTSRB/... 和 ./GT-final_test.csv
mkdir -p MLproject/data/GTSRB/raw
tar -xzf transfer/a_gtsrb_raw.tgz -C MLproject/data/GTSRB/raw
```

## 第 6 步（Spark）：校验数据与权重

```bash
cd ~/ds4023_a/MLproject

# 数量必须完全一致
find data/GTSRB/raw/GTSRB/Final_Training/Images -name '*.ppm' | wc -l    # 39209
find data/GTSRB/raw/GTSRB/Final_Test/Images -name '*.ppm' | wc -l         # 12630

# 标注与划分的指纹
sha256sum data/GTSRB/raw/GT-final_test.csv     # 58a660fa…1c12f
sha256sum results/pilot_split.csv              # de914a91…aa5359

# 官方初始化权重（不在仓库里，需要自己下）
curl -fL https://download.pytorch.org/models/resnet18-f37072fd.pth \
  -o assets/weights/resnet18-f37072fd.pth
sha256sum assets/weights/resnet18-f37072fd.pth # f37072fd…0e07ec
```

如果 Spark 上不去 pytorch.org，就在 Mac 上下好后 `scp` 过来（46 MB），校验值同上。

## 第 7 步（Spark）：建镜像并检查

```bash
cd ~/ds4023_a/MLproject
bash A_SPARK.sh setup
```

- 第一次会从 `nvcr.io` 拉 `nvidia/pytorch:25.11-py3`（约 20 GB，慢），然后在本地构建 `ds4023-c-resnet18-spark:25.11`。
- 拉取提示需要认证时，在 Spark 本机执行 `docker login nvcr.io`（用你自己的 NGC 账号，不要把凭据发给别人）。
- 构建完会自动跑 `check`：**真实 batch 128 的前向/反向/优化器一步**（报峰值内存和单步耗时、估算每次训练分钟数）+ 数据预算审计 + 四组回归测试。

看到 `check` 全绿再往下。

## 第 8 步（Spark）：单轮试跑

```bash
cd ~/ds4023_a/MLproject
bash A_SPARK.sh smoke
```

`mixed × 种子 42 × 1 轮`，写入 `results/a_spark_smoke_v1/`。不计入正式结果。

## 第 9 步（Spark）：正式九次训练

```bash
tmux new -s a-train
cd ~/ds4023_a/MLproject
bash A_SPARK.sh train 2>&1 | tee -a ~/ds4023_a/logs/train.log
```

- 矩阵：`raw` 输入 × `repeat_clean / targeted_motion / mixed` × 种子 `42/43/44`，共 9 次，每次 10 轮（350 次更新）。
- 一个容器顺序跑完 9 次，中途不要关 tmux；`Ctrl-B D` 脱离，`tmux attach -t a-train` 回来。
- 中断后：`bash A_SPARK.sh resume`（只跳过已完整通过核验的运行，半成品会被拒绝，那就换 `A_OUTPUT=results/a_spark_v2` 重跑）。

## 第 10 步（Spark）：核验、评估、汇总

```bash
cd ~/ds4023_a/MLproject
bash A_SPARK.sh verify     # 应为 Verified 9 completed official ResNet18 runs
bash A_SPARK.sh eval13     # 开发集十三条件评估，不重新训练
bash A_SPARK.sh report     # 表格、配对差值、强度曲线 → report/a_part_spark/
```

跑完应该有的数量（少一个都说明产物不完整）：

| 产物 | 数量 |
|---|---|
| `results/a_spark_v1/runs/` | 9 个目录 |
| 训练指标行（5 条件 × 9） | 45 |
| 训练逐图预测（4,200 × 9） | 37,800 |
| 训练特征文件（6 × 9） | 54 |
| 评估指标行（13 × 9） | 117 |
| 评估逐图预测（13 × 840 × 9） | 98,280 |
| 评估特征文件（13 × 9） | 117 |

## 第 11 步（Spark）：打包带回

```bash
cd ~/ds4023_a
tar -czf a_spark_results.tgz -C MLproject \
  results/a_spark_v1 results/a_eval13_spark_v1 results/a_spark_environment \
  results/a_spark_smoke_v1 report/a_part_spark
ls -lh a_spark_results.tgz
```

在 Mac 上拉回本地：

```bash
scp jeremy@spark-d01d:~/ds4023_a/a_spark_results.tgz ~/Desktop/
```

再从那台 Mac 传回 Windows（用你的远控/网盘），放到 `C:\Users\Weisifengbuxi\Desktop\ML project\` 下，我接着做结果分析和报告。

## 第 12 步（后置）：官方七类测试

按协议要等全组方案冻结后才打开测试分数：

```bash
cd ~/ds4023_a/MLproject
bash A_SPARK.sh freeze     # 写出 results/a_test7_freeze.json
bash A_SPARK.sh test7      # 独立测试评估
```

---

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `A_SPARK.sh env` 里 Docker FAIL | 当前用户不在 docker 组或守护进程未起 | 让机主配好 Docker 访问，见 DGX_SPARK.md；这步不过不要继续 |
| 拉 `nvcr.io` 认证失败 | NGC 需要登录 | Spark 上 `docker login nvcr.io` |
| 训练特别慢 | vLLM 占了 88 GB、GPU 95% | 和机主确认能否停掉；`check` 的耗时估计已经包含了当时的负载 |
| `Batch 128 does not fit` | 统一内存被别的进程吃掉 | 释放 vLLM；**不要**自己降 batch 或开 AMP，那会破坏全组协议 |
| `Study code fingerprints differ` | 训练后改过源码 | 换新目录（`A_OUTPUT=results/a_spark_v2`）重跑，不要手改指纹 |
| `Run was trained on different generated pixels` | 训练与评估不在同一环境 | 在同一镜像里重跑评估 |
| `Output already exists` | 目录已存在，程序拒绝覆盖 | 换目录名 |
| 想手工排查 | — | `bash A_SPARK.sh shell` 进同一个容器，路径是 `/workspace` |
| `RuntimeError: … does not have a deterministic implementation` | 确定性算子在这张卡/这个构建上缺失 | 记下完整报错发我，**不要**擅自改成 `warn_only` 后继续跑正式结果 |
| 磁盘不够 | 镜像 20 GB + 九个运行约 5 GB | `df -h ~`，先清掉旧镜像 |

## 两个必须记住的边界

1. **不要用 `A_LINUX.sh setup`**。那是 x86_64 的 pip 路线，在 aarch64 上装不上或装成 CPU 版。Spark 只走 `A_SPARK.sh`，环境由 NVIDIA 的 ARM64 镜像提供。
2. **Spark 的结果不能和 Windows 的 15 次结果混进同一个三种子均值**。两边的框架构建、硬件和浮点执行都不同；`data_fingerprint` 也可能不同。报告里要分开写两套结果。
