# MLproject

DS 4023 交通标志鲁棒性研究。项目使用 GTSRB 七类限速标志，比较训练增强策略、CLAHE 单分支与双视图融合。当前 C 部分使用 Torchvision 官方 ImageNet 预训练 ResNet18，共享编码器导出 512 维特征。正式矩阵为五组配置各三个种子，共 15 次。实验设置见 [训练协议](PROTOCOL.md)。

团队成员在各自的分支提交改动，通过 Pull Request 合入 `main`。[贡献指南](CONTRIBUTING.md)

## 在 DGX Spark 运行

```bash
git clone https://github.com/yinianqingxian/MLproject.git
cd MLproject
```

先准备数据和初始化权重。已有 `DS4023_TrafficSign_ResNet18_Spark_20261117.zip` 时，可按 [解压与训练步骤](ResNet18_Spark_完整操作步骤.md) 直接在解压目录运行。若从本仓库运行，复制解压项目中的 `data/GTSRB/raw/` 和 `assets/weights/resnet18-f37072fd.pth` 到本仓库对应路径。

也可以从 [GTSRB 官方归档](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html) 下载 `GTSRB_Final_Training_Images.zip`、`GTSRB_Final_Test_Images.zip` 和 `GTSRB_Final_Test_GT.zip`。整理为以下结构，保留仓库中的 `results/pilot_split.csv`：

```text
data/GTSRB/raw/
├── GTSRB/Final_Training/Images/<类别目录>/
├── GTSRB/Final_Test/Images/
└── GT-final_test.csv
```

```bash
curl -fL https://download.pytorch.org/models/resnet18-f37072fd.pth \
  -o assets/weights/resnet18-f37072fd.pth
printf '%s\n' 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec  assets/weights/resnet18-f37072fd.pth' | sha256sum -c -
```

权重校验通过后，在 Spark 项目根目录执行。每一步成功后再执行下一步。

```bash
bash SPARK.sh setup
bash SPARK.sh tests
bash SPARK.sh smoke
bash SPARK.sh train
bash SPARK.sh verify
bash SPARK.sh report
```

训练结果默认写入 `results/study_resnet18_spark_v1/`，报告写入 `report/c_part_resnet18_spark/C_REPORT.md`。环境检查与失败恢复见 [DGX Spark 说明](DGX_SPARK.md)。原始数据、初始化文件、大型训练产物与离线安装材料由训练机器或独立训练包保存。

## 已有结果

- 历史 SmallCNN 的 15 次开发验证结果，含报告、图表和汇总指标。 [历史 C 报告](report/c_part/C_REPORT.md)
- B 的 12 组 ResNet18 单分支汇总与输入一致性检查。正式配对分析需要先对齐输入像素。 [B/C 交接审查](results/b_handoff_review_20261122/REVIEW.md)
- C 的 15 次正式 ResNet18 开发验证实验已在 Windows NVIDIA GeForce RTX 4060 Ti 上跑完并通过核验。 [正式 ResNet18 报告](report/c_part_resnet18_rtx4060ti/C_REPORT.md) · [项目进度](MASTER.md)
- C 的模型和特征可用于分组复核与聚类。 [交接说明](handoffs/C_RESNET18_HANDOFF.md) · [单路与双路推理计时](results/c_inference_cost_rtx4060ti_20261123/README.md)
