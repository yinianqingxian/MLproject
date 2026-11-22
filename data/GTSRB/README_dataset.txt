GTSRB German Traffic Sign Recognition Benchmark

官方来源 https://github.com/houbensebastian/GermanTrafficSignBenchmarks
数据来自官方最终训练图、最终测试图与测试标签三个 ZIP。

GTSRB/Final_Training/Images/ 是完整训练集，含 43 类和 39209 张 PPM 图片。
每个类别目录附原始 GT-xxxxx.csv，包含类别、图片尺寸和标志 ROI。
GTSRB/Final_Test/Images/ 是 12630 张官方测试图。
GT-final_test.csv 是官方测试标签，按 Filename 与测试图连接。
results/pilot_split.csv 是七类限速牌预实验清单，image 路径相对于 data/GTSRB/raw/。
其中 train 为 2940 张、98 条轨迹，validation 为 840 张、28 条轨迹。
清单来自官方训练集，轨迹 ID 由类别和图片文件名前缀组成。
本次预实验的增强和验证扰动由实验脚本生成。
