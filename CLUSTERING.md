# 用同一模型的特征做七类聚类

用已经完成的模型产物，在 CPU 上直接对池化后的 512 维特征做 K-means。

目标是检查模型学到的表示能否自然分成七类。先在干净训练特征上拟合 KMeans，再预测验证特征的簇；只用**训练标签**确定“簇编号对应哪个类别”。验证标签只用于最后评分。

## 1. 先选对产物

每次只选一个已完成 run 的 `features/train_clean.npz`，搭配**同一个 run** 的 `features/validation_clean.npz`，也可换成 `validation_motion.npz`、`validation_noise.npz`、`validation_dark.npz` 或 `validation_jpeg.npz`。项目原始七类标签是 `1, 2, 3, 4, 5, 7, 8`；正常划分为 2940 张训练图、840 张验证图，训练和验证的 track 也应互不重叠。

脚本使用 NPZ 的 `features`：单路时是该路表示，双路时是融合后的表示。`raw_features`、`enhanced_features` 属于分支分析，不应和最终融合特征混着比较。旧输出没有 `pooling` 配置时按历史 GAP 处理；给旧目录改名并不会得到 AvgMax 特征。

手头已有同一 run 的特征时，直接进入下一步。需要新导出 AvgMax 特征时，在仓库根目录用下面的命令训练增强单路和空间门控双路。数据及 ImageNet 初始化权重的路径见 [项目说明](README.md)。

```bash
python experiments/run_resnet_study.py --pooling avgmax --augmentations mixed --inputs denoised_clahe dual_spatial --output-dir results/resnet18_avgmax --device cuda
```

AvgMax 在每个通道取全局平均与全局最大响应的等权平均。单路直接对该路特征图池化。整图双路分别池化后融合，空间双路按位置融合后再池化。四个残差阶段和七分类头的维数照旧。

## 2. 复制这一整段到 notebook cell

需要 NumPy、SciPy、scikit-learn 和 threadpoolctl。在仓库根目录运行，把 `RUN` 改成要分析的已完成 run。也可以把整段保存为 Python 脚本运行。

```python
from pathlib import Path
import hashlib
import json
import numpy as np
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

RUN = Path("results/resnet18_avgmax/runs/mixed_dual_spatial_seed42").resolve()
VALIDATION = "validation_clean"  # 可改成 validation_motion/noise/dark/jpeg
KMEANS_SEED = 42                # 与模型训练 seed 分开记录
CLASSES = np.array([1, 2, 3, 4, 5, 7, 8], dtype=np.int64)

if VALIDATION not in {"validation_clean", "validation_motion", "validation_noise",
                      "validation_dark", "validation_jpeg"}:
    raise ValueError("请选择本 run 已导出的 validation 条件")
completion = json.loads((RUN / "completion.json").read_text(encoding="utf-8"))
if completion.get("status") != "complete":
    raise ValueError("run 尚未完成；不要分析中途或损坏的产物")

# 验证三份输入确实属于这次完成产物；不加载 model.pt/pickle。
for relative in ("config.json", "features/train_clean.npz",
                 f"features/{VALIDATION}.npz"):
    expected = completion.get("artifacts", {}).get(relative)
    actual = hashlib.sha256((RUN / relative).read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError(f"完成清单缺失或 SHA256 不匹配：{relative}")
config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
if config.get("classes") != CLASSES.tolist():
    raise ValueError("这不是本教程的七类配置")

def load_features(name):
    with np.load(RUN / "features" / f"{name}.npz", allow_pickle=False) as z:
        required = {"features", "y_true", "image", "track_id", "y_pred"}
        if not required.issubset(z.files):
            raise ValueError(f"{name} 缺少字段：{required - set(z.files)}")
        arrays = {key: z[key].copy() for key in required}
    x, y = arrays["features"], arrays["y_true"]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f"{name} 特征不是有限的二维矩阵")
    if y.shape != (len(x),) or y.dtype.kind not in "iu":
        raise ValueError(f"{name} 标签形状或类型错误")
    if set(np.unique(y)) != set(CLASSES):
        raise ValueError(f"{name} 未包含预期七类")
    for key in ("image", "track_id", "y_pred"):
        if arrays[key].shape != y.shape:
            raise ValueError(f"{name}/{key} 与样本数不一致")
    for key in ("image", "track_id"):
        if arrays[key].dtype.kind not in "US" or any(not str(v).strip() for v in arrays[key]):
            raise ValueError(f"{name}/{key} 不能用于划分核验")
    return arrays

train, val = load_features("train_clean"), load_features(VALIDATION)
if (len(train["features"]), len(val["features"])) != (2940, 840):
    raise ValueError("样本数不是本项目冻结划分的 2940/840")
if train["features"].shape[1] != 512 or val["features"].shape[1] != 512:
    raise ValueError("本实验要求训练和验证都是 512 维特征")
def identities(array):
    return {str(v).replace("\\", "/").casefold() for v in array}
for key in ("image", "track_id"):
    if identities(train[key]) & identities(val[key]):
        raise ValueError(f"训练/验证 {key} 重叠，停止分析")
for split in (train, val):
    if len(identities(split["image"])) != len(split["image"]):
        raise ValueError("同一划分内含重复图像")

# 每张图的特征独立做 L2 归一化，簇中心只在训练集拟合。
with threadpool_limits(limits=2):
    x_train = normalize(train["features"])
    x_val = normalize(val["features"])
    km = KMeans(n_clusters=7, init="k-means++", n_init=15,
                max_iter=300, algorithm="lloyd", random_state=KMEANS_SEED)
    train_cluster = km.fit_predict(x_train)
    val_cluster = km.predict(x_val)
if len(np.unique(train_cluster)) != 7:
    raise ValueError("不足七个非空簇；先检查特征是否退化")

# 匈牙利匹配：最大化训练集的“簇→类别”一对一匹配正确数。
# 此处绝不使用 val['y_true'] 来挑选映射。
counts = np.array([[np.sum((train_cluster == c) & (train["y_true"] == y))
                    for y in CLASSES] for c in range(7)])
cluster_ids, class_indices = linear_sum_assignment(-counts)
mapping = {int(c): int(CLASSES[j]) for c, j in zip(cluster_ids, class_indices)}
mapped_val = np.array([mapping[int(c)] for c in val_cluster])
cluster_f1 = f1_score(val["y_true"], mapped_val, labels=CLASSES,
                      average="macro", zero_division=0)
head_f1 = f1_score(val["y_true"], val["y_pred"], labels=CLASSES,
                   average="macro", zero_division=0)
print({"run_id": config.get("run_id", RUN.name), "model_seed": config.get("seed"),
       "input": config.get("input"), "fusion": config.get("fusion", completion.get("fusion")),
       "pooling": config.get("pooling", "gap"), "condition": VALIDATION,
       "feature_dim": int(train["features"].shape[1]), "kmeans_seed": KMEANS_SEED,
       "n_init": 15, "normalization": "L2 per image", "sklearn": sklearn.__version__,
       "ARI": adjusted_rand_score(val["y_true"], val_cluster),
       "NMI": normalized_mutual_info_score(val["y_true"], val_cluster,
                                            average_method="arithmetic"),
       "cluster_macro_f1": float(cluster_f1), "original_head_macro_f1": float(head_f1),
       "delta_f1_percentage_points": float(100 * (cluster_f1 - head_f1)),
       "train_cluster_to_class": mapping})
```

## 3. 看什么，怎么汇总

ARI、NMI 直接比较验证标签与原始簇编号，不需要类别映射。ARI 越接近 1 越好，接近 0 表示与随机分组相近，也可能为负；NMI 范围为 0–1，越高表示分组与类别关系越强。它们衡量表示的可聚类程度，和分类头的 Macro-F1 含义不同。

`cluster_macro_f1` 是先按训练标签固定簇映射、再在验证集算的七类等权 F1；`original_head_macro_f1` 是同一 NPZ 保存的原分类头结果。两者使用同一批验证图和标签，可直接看差值。脚本 F1 用 0–1 表示，乘 100 得百分数；`delta_f1_percentage_points` 已经是百分点。聚类用到了训练标签做解释，因此这里是“无监督聚类 + 训练标签匹配”，不能称为完全不使用标签的分类器。

先固定 `KMEANS_SEED=42`，分别对模型 seed 42、43、44 的 run 独立运行；记录每个 seed 的指标，再报告三次均值与标准差。不要把不同模型的特征行拼在一起拟合一个 KMeans，它们的特征坐标系并不相同。若检查聚类初始化稳定性，可对每个模型用同一组额外 KMeans seed 重复；这些重复不能当成更多独立模型 seed，也不能挑最高验证分数来汇报。

比较 GAP 与 AvgMax 时，应固定输入模式、单双路/门控方式、训练 seed、数据划分、验证条件及这段聚类配置；同时报告实际 `feature_dim`。维数不同会改变聚类距离和容量，不能只写“都是 ResNet18”。要比较 12 种干扰平均，必须给两组完全相同的 12 个条件分别评分再求平均；上面列出的四个旧 `validation_*.npz` 只是四种导出条件，不能冒充完整 12 条件。只在一两个条件上提高，也不能推出总体更好。

遇到文件缺失、SHA256 不符、字段错误或样本/track 重叠，先停止并让产物提供者核对对应 run。不要通过删除检查、借用别的 seed 的验证 NPZ 或使用验证标签重新匹配簇来补齐结果。
