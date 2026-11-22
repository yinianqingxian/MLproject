# 官方 ResNet18 初始化权重

文件 `resnet18-f37072fd.pth` 来自 PyTorch 官方下载站，对应 torchvision 的 `ResNet18_Weights.IMAGENET1K_V1`：

- 来源：[官方权重](https://download.pytorch.org/models/resnet18-f37072fd.pth)
- SHA256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- API 依据：[torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

这是 ImageNet 初始化，不是本项目已经训练完成的 ResNet 模型。程序在使用前核对完整指纹，正式训练与旧 SmallCNN 结果分开保存。权重随新版 Spark 训练包携带，仍需联网拉取 NVIDIA 容器与缺失依赖。
