# A 十三条件开发集汇总

数据来源：`results/a_local_v1`，9 次运行，每个运行 840 张开发集图像、13 个冻结条件。

表内为三个训练种子的均值 ± 样本标准差（`ddof=1`），单位为 Macro-F1 百分比。标准差只描述三个种子的训练随机性，不能当作置信区间。

## 全部十三条件

| 条件 | 强度 | mixed | repeat_clean | targeted_motion |
|---|---|---|---|---|
| clean | none | 96.07 ± 3.06 | 97.15 ± 0.62 | 95.42 ± 1.82 |
| motion | low | 93.92 ± 2.78 | 93.07 ± 1.71 | 93.30 ± 1.78 |
| motion | medium | 85.44 ± 5.07 | 63.58 ± 2.35 | 87.30 ± 1.63 |
| motion | high | 72.29 ± 9.05 | 46.18 ± 3.49 | 80.39 ± 2.07 |
| noise | low | 93.43 ± 3.67 | 89.28 ± 3.80 | 90.61 ± 0.47 |
| noise | medium | 86.82 ± 5.77 | 66.43 ± 6.89 | 69.20 ± 8.37 |
| noise | high | 77.81 ± 7.58 | 45.93 ± 5.10 | 52.09 ± 9.41 |
| dark | low | 96.51 ± 2.06 | 97.94 ± 0.48 | 93.26 ± 2.82 |
| dark | medium | 93.74 ± 0.85 | 97.30 ± 0.28 | 83.86 ± 8.64 |
| dark | high | 82.57 ± 3.35 | 87.47 ± 2.79 | 52.78 ± 6.32 |
| jpeg | low | 88.61 ± 8.20 | 86.73 ± 2.14 | 87.19 ± 3.56 |
| jpeg | medium | 73.20 ± 10.37 | 64.88 ± 3.81 | 66.52 ± 8.71 |
| jpeg | high | 52.50 ± 9.06 | 41.51 ± 6.78 | 47.19 ± 10.82 |

## 干扰族平均与 12 干扰条件

每族取三档强度的均值；`all_interference` 为 12 个干扰条件的均值，`worst` 为其中最差条件。

| 配置 | 族 | 均值 Macro-F1 | 最差 | 相对干净图下降 |
|---|---|---|---|---|
| mixed | motion | 83.88 | 72.29 | +12.19 |
| mixed | noise | 86.02 | 77.81 | +10.05 |
| mixed | dark | 90.94 | 82.57 | +5.13 |
| mixed | jpeg | 71.43 | 52.50 | +24.64 |
| mixed | all_interference | 83.07 | 52.50 | +13.00 |
| repeat_clean | motion | 67.61 | 46.18 | +29.54 |
| repeat_clean | noise | 67.21 | 45.93 | +29.94 |
| repeat_clean | dark | 94.23 | 87.47 | +2.91 |
| repeat_clean | jpeg | 64.37 | 41.51 | +32.78 |
| repeat_clean | all_interference | 73.36 | 41.51 | +23.79 |
| targeted_motion | motion | 87.00 | 80.39 | +8.43 |
| targeted_motion | noise | 70.63 | 52.09 | +24.79 |
| targeted_motion | dark | 76.63 | 52.78 | +18.79 |
| targeted_motion | jpeg | 66.96 | 47.19 | +28.46 |
| targeted_motion | all_interference | 75.31 | 47.19 | +20.12 |

## 配对差值（同种子）

差值 = 目标配置 − 基线配置，逐种子相减后再求均值 ± 样本标准差；`同向种子` 是差值为正的种子数，用于判断稳定性，不能单独当作显著性检验。

| 比较 | 条件 | 强度 | 差值均值 | 差值 SD | 同向种子 |
|---|---|---|---|---|---|
| targeted_motion - repeat_clean | clean | none | -1.73 | 1.22 | 0/3 |
| targeted_motion - repeat_clean | motion | medium | +23.72 | 3.83 | 3/3 |
| targeted_motion - repeat_clean | noise | medium | +2.77 | 10.63 | 2/3 |
| targeted_motion - repeat_clean | dark | medium | -13.43 | 8.81 | 0/3 |
| targeted_motion - repeat_clean | jpeg | medium | +1.64 | 11.74 | 2/3 |
| mixed - repeat_clean | clean | none | -1.08 | 2.82 | 1/3 |
| mixed - repeat_clean | motion | medium | +21.86 | 2.75 | 3/3 |
| mixed - repeat_clean | noise | medium | +20.40 | 11.71 | 3/3 |
| mixed - repeat_clean | dark | medium | -3.56 | 0.93 | 0/3 |
| mixed - repeat_clean | jpeg | medium | +8.32 | 10.74 | 2/3 |
| mixed - targeted_motion | clean | none | +0.65 | 2.37 | 1/3 |
| mixed - targeted_motion | motion | medium | -1.86 | 6.58 | 1/3 |
| mixed - targeted_motion | noise | medium | +17.63 | 13.20 | 3/3 |
| mixed - targeted_motion | dark | medium | +9.87 | 9.27 | 3/3 |
| mixed - targeted_motion | jpeg | medium | +6.68 | 17.88 | 2/3 |

完整十三条件的配对差值见 `tables/dev13_paired.csv`。
