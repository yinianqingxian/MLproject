# GTSRB 七类限速牌训练与聚类

`train_and_cluster.py` 是带英文注释的单文件实现，直接使用官方 ResNet18 和原生全局平均池化（GAP）。

| 开关 | 开启 | 关闭 |
|---|---|---|
| `bilateral` | 双边滤波。两项图像处理都开时，先滤波再做 CLAHE | 跳过滤波 |
| `clahe` | 对 LAB 亮度通道做 CLAHE | 跳过 CLAHE |
| `gate` | 原图和处理图分别通过共享编码器，由标量门控融合。图像处理都关时，两路都是原图 | 处理后的图走单路。图像处理都关时使用原图 |
| `a2` | A2 混合增强，追加运动模糊、噪声和变暗样本各 490 张 | A0，重复同一批 1,470 张原图，保持训练预算一致 |
| `cluster` | 用训练后的特征做 KMeans，计算 ARI、NMI 和同簇率 | 跳过 KMeans，仍输出分类结果和 PCA |

## 准备

1. 将本地七类 GTSRB 原始训练图像放在仓库的 `data/GTSRB/raw/`，或把 `--data-root` 指向包含 `GTSRB/Final_Training/Images/` 的文件夹。命令使用仓库已有的 `results/pilot_split.csv`。冻结划分包含 2,940 张训练图和 840 张开发验证图。
2. 准备与机器匹配的 PyTorch 环境。已核验的组合为 Python 3.12、Torch 2.5.1（CUDA 12.4）和 Torchvision 0.20.1。在这个环境里执行 `python -m pip install -r requirements.txt`。
3. 将官方 ImageNet 初始化权重保存为仓库的 `assets/weights/resnet18-f37072fd.pth`。下载地址和 SHA-256 在[官方权重说明](../../assets/weights/README.md)中。脚本在训练前核对完整指纹。

## 运行

在交付目录执行以下命令。加 `--select` 会在读取数据前显示编号和五个开关的状态。

```bash
python train_and_cluster.py --data-root ../../data/GTSRB/raw --split ../../results/pilot_split.csv --imagenet-weights ../../assets/weights/resnet18-f37072fd.pth --output ../results_selected --select
```

输入 `1, 4, 7` 这类编号来多选，也可以用空格分隔。按回车选中菜单里的全部组合，输入 `q` 退出。脚本会显示选中的结果组合数和实际训练次数。

省略 `--select` 时，默认直接运行五个开关的全部 32 个组合。KMeans 只分析训练后的特征，开关配对共用同一份权重。因此共有 16 种训练配置，每种运行 42、43、44 三个种子，合计 48 次训练，每次 10 轮。

用 `--no-a2` 等开关筛选组合，菜单编号按筛选后的列表显示。加 `--seeds 42` 只跑一个种子，加 `--device cpu` 使用 CPU。每次运行选一个新的输出目录。

用 `--group custom` 运行单个手动配置。默认值集中在脚本顶部的 `CUSTOM_SWITCHES`，也可以在命令行逐个开关。以下命令运行仅双边滤波的单路模型，使用 A2 并关闭 KMeans。

```bash
python train_and_cluster.py --data-root ../../data/GTSRB/raw --split ../../results/pilot_split.csv --imagenet-weights ../../assets/weights/resnet18-f37072fd.pth --output ../results_bilateral --group custom --bilateral --no-clahe --no-gate --a2 --no-cluster
```

## 产物

- `runs/<run_id>/model.pt` 为末轮权重，同目录保存干净训练特征、13 个验证条件的特征与预测，以及 PCA 坐标。
- `combinations.csv` 每行对应一种开关组合，列出 13 条件及 12 个受扰条件的平均准确率和 Macro-F1。开启 KMeans 的行也记录 13 条件平均 ARI、NMI 和同簇率。组合名中 `b/c/g/a/k` 对应表中的五个开关，`0` 代表关闭，`1` 代表开启。
- `classification.csv` 记录每个训练配置、种子和条件的分类结果。开启聚类时生成 `clustering.csv`，记录对应的逐条件聚类结果。
- `summary.json` 汇总全 13 条件和 12 个受扰条件。多种子记录均值和样本标准差。
- `conditions.json` 保存 13 个条件的像素 SHA-256。`pca_seed42.png` 默认展示前 3 种训练配置在干净和四类高档干扰下的对照，每个训练都导出完整的 PCA 坐标。

KMeans 和 PCA 都用干净训练特征拟合。改变软硬件环境时，可先核对 13 个条件的像素指纹，再比较训练结果。
