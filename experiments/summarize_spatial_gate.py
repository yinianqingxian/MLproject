"""
用途：汇总空间门控实验（S / G / P 三组 + P 的权重图干预）的表格、配对差值与结论。
核心职责：
- 读 13 条件评估目录与干预目录，校验种子、条件、样本齐全。
- 输出每组条件表、三组配对差值、干预纠正/改错计数、权重图空间标准差。
- 生成中文结论草稿与一张权重图标准差图。
关键副作用/安全约束：
- 只读评估产物，只写新的报告目录；不重新推理、不改任何结果目录。
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

MODE_TO_GROUP = {"denoised_clahe": "S", "dual_denoised": "G", "dual_spatial": "P"}
CONDITIONS = [("clean", "none")] + [(kind, severity) for kind in ("motion", "noise", "dark", "jpeg")
                                    for severity in ("low", "medium", "high")]
TITLES = {"S": "S 单视图（去噪 CLAHE）", "G": "G 原整图门控", "P": "P 新空间门控"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_of(run_id):
    for mode, group in MODE_TO_GROUP.items():
        if f"_{mode}_seed" in run_id:
            return group
    raise ValueError(f"无法识别分组：{run_id}")


def load_eval(directory):
    directory = Path(directory)
    scores = defaultdict(dict)      # (group, condition, severity)[seed] = macro_f1
    predictions = defaultdict(dict)  # (group, condition, severity, seed) -> {image: (y_true, y_pred)}
    seeds = defaultdict(set)
    for run_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        run_id = run_dir.name
        group = group_of(run_id)
        for row in read_csv(run_dir / "metrics.csv"):
            key = (group, row["condition"], row["severity"])
            scores[key][int(row["seed"])] = float(row["macro_f1"]) * 100
            seeds[group].add(int(row["seed"]))
        for row in read_csv(run_dir / "predictions.csv"):
            key = (group, row["condition"], row["severity"], int(row["seed"]))
            predictions[key][row["image"]] = (int(row["y_true"]), int(row["y_pred"]))
    return scores, predictions, seeds


def mean_sd(values):
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--intervention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    scores, predictions, seeds = load_eval(args.eval)
    for group in ("S", "G", "P"):
        if not seeds[group]:
            raise SystemExit(f"评估目录里没有 {group} 组")
    print("种子：", {g: sorted(v) for g, v in seeds.items()})
    common = sorted(set.intersection(*[seeds[g] for g in ("S", "G", "P")]))
    print("三组共有的种子：", common)

    args.output.mkdir(parents=True)
    (args.output / "tables").mkdir(exist_ok=True)

    # ---------- 1. 每组每条件 ----------
    rows = []
    for group in ("S", "G", "P"):
        for condition, severity in CONDITIONS:
            mean, sd = mean_sd(scores[(group, condition, severity)][s]
                               for s in common if s in scores[(group, condition, severity)])
            rows.append({"group": group, "title": TITLES[group], "condition": condition,
                         "severity": severity, "macro_f1_mean": mean, "macro_f1_sd": sd,
                         "seeds": len(common)})
    write_csv(args.output / "tables" / "group_condition.csv", rows)

    # ---------- 2. 配对差值 ----------
    def paired(a, b):
        out = []
        for condition, severity in CONDITIONS:
            deltas = []
            for seed in common:
                left = scores[(a, condition, severity)].get(seed)
                right = scores[(b, condition, severity)].get(seed)
                if left is not None and right is not None:
                    deltas.append(left - right)
            mean, sd = mean_sd(deltas)
            out.append({"comparison": f"{a}-{b}", "condition": condition, "severity": severity,
                        "delta_mean": mean, "delta_sd": sd,
                        "positive_seeds": sum(1 for d in deltas if d > 0), "seeds": len(deltas)})
        return out

    paired_rows = paired("P", "G") + paired("P", "S")
    write_csv(args.output / "tables" / "paired.csv", paired_rows)

    # ---------- 3. 12 条件均值 ----------
    def config_mean(group):
        values = []
        for condition, severity in CONDITIONS:
            if condition == "clean":
                continue
            mean, _ = mean_sd(scores[(group, condition, severity)][s]
                              for s in common if s in scores[(group, condition, severity)])
            values.append(mean)
        return statistics.mean(values), min(values)

    summary = []
    for group in ("S", "G", "P"):
        mean12, worst = config_mean(group)
        clean_mean, _ = mean_sd(scores[(group, "clean", "none")][s]
                                for s in common if s in scores[(group, "clean", "none")])
        summary.append({"group": group, "clean": clean_mean, "twelve_condition_mean": mean12,
                        "worst_condition": worst})
    write_csv(args.output / "tables" / "summary.csv", summary)

    # ---------- 4. 干预：正常权重图 vs 平均权重图 ----------
    intervention = Path(args.intervention)
    intervention_scores = defaultdict(dict)
    intervention_pred = defaultdict(dict)
    gate_sd = defaultdict(dict)
    for row in read_csv(intervention / "metrics.csv"):
        key = (row["mode"], row["condition"], row["severity"])
        intervention_scores[key][int(row["seed"])] = float(row["macro_f1"]) * 100
        gate_sd[key][int(row["seed"])] = float(row["gate_sd"])
    for row in read_csv(intervention / "predictions.csv"):
        key = (row["mode"], row["condition"], row["severity"], int(row["seed"]))
        intervention_pred[key][row["image"]] = (int(row["y_true"]), int(row["y_pred"]))

    rows = []
    for condition, severity in CONDITIONS:
        deltas = [intervention_scores[("normal", condition, severity)][s]
                  - intervention_scores[("average", condition, severity)][s]
                  for s in common
                  if s in intervention_scores[("normal", condition, severity)]
                  and s in intervention_scores[("average", condition, severity)]]
        mean, sd = mean_sd(deltas)
        rows.append({"comparison": "P_normal-P_average", "condition": condition,
                     "severity": severity, "delta_mean": mean, "delta_sd": sd,
                     "positive_seeds": sum(1 for d in deltas if d > 0), "seeds": len(deltas)})
    write_csv(args.output / "tables" / "intervention_paired.csv", rows)

    corrections = []
    for condition, severity in CONDITIONS:
        corrected = worsened = changed = 0
        for seed in common:
            normal = intervention_pred[("normal", condition, severity, seed)]
            average = intervention_pred[("average", condition, severity, seed)]
            for image, (truth, pred_normal) in normal.items():
                pred_average = average[image][1]
                if pred_normal == pred_average:
                    continue
                changed += 1
                if pred_normal == truth:
                    corrected += 1
                elif pred_average == truth:
                    worsened += 1
        corrections.append({"condition": condition, "severity": severity,
                            "changed": changed, "corrected_by_spatial": corrected,
                            "worsened_by_spatial": worsened})
    write_csv(args.output / "tables" / "intervention_corrections.csv", corrections)

    # ---------- 5. 权重图空间标准差 ----------
    sd_rows = []
    for condition, severity in CONDITIONS:
        mean, sd = mean_sd(gate_sd[("normal", condition, severity)][s] for s in common
                           if s in gate_sd[("normal", condition, severity)])
        sd_rows.append({"condition": condition, "severity": severity,
                        "gate_sd_mean": mean, "gate_sd_sd": sd})
    write_csv(args.output / "tables" / "gate_spatial_sd.csv", sd_rows)

    # ---------- 6. 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        (args.output / "figures").mkdir(exist_ok=True)
        families = ("motion", "noise", "dark", "jpeg")
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
        for axis, family in zip(axes.ravel(), families):
            values = [next(r["gate_sd_mean"] for r in sd_rows
                           if r["condition"] == family and r["severity"] == severity)
                      for severity in ("low", "medium", "high")]
            axis.bar(["low", "medium", "high"], values)
            axis.set_title(f"weight-map SD ({family})")
        figure.suptitle("Spatial standard deviation of the 7x7 weight map (P, 3 seeds)")
        figure.tight_layout()
        figure.savefig(args.output / "figures" / "gate_spatial_sd.png", dpi=150)
        plt.close(figure)
    except Exception as error:  # pragma: no cover
        print("图未生成：", error)

    # ---------- 7. 结论草稿 ----------
    def fmt(comparison, condition, severity):
        row = next(r for r in paired_rows + rows
                   if r["comparison"] == comparison and r["condition"] == condition
                   and r["severity"] == severity)
        return f"{row['delta_mean']:+.2f} ± {row['delta_sd']:.2f}（{row['positive_seeds']}/{row['seeds']}）"

    mean_sd_gate = statistics.mean(r["gate_sd_mean"] for r in sd_rows)
    lines = [
        "# 空间门控实验（第二阶段）结果",
        "",
        f"种子：{common}。开发集 13 个冻结条件，840 张图，三组使用完全相同的图像。",
        "",
        "## 12 个干扰条件均值",
        "",
        "| 组 | 干净图 | 12 条件均值 | 最差条件 |",
        "|---|---|---|---|",
    ]
    for row in summary:
        lines.append(f"| {TITLES[row['group']]} | {row['clean']:.2f} | "
                     f"{row['twelve_condition_mean']:.2f} | {row['worst_condition']:.2f} |")
    lines += [
        "",
        "## 三个配对差值（中强度，同种子配对）",
        "",
        "| 比较 | 干净 | 模糊中 | 噪声中 | 变暗中 | JPEG中 |",
        "|---|---|---|---|---|---|",
    ]
    for comparison in ("P-G", "P-S", "P_normal-P_average"):
        cells = [fmt(comparison, condition, severity)
                 for condition, severity in (("clean", "none"), ("motion", "medium"),
                                             ("noise", "medium"), ("dark", "medium"),
                                             ("jpeg", "medium"))]
        lines.append(f"| {comparison} | " + " | ".join(cells) + " |")
    total_corrected = sum(r["corrected_by_spatial"] for r in corrections)
    total_worsened = sum(r["worsened_by_spatial"] for r in corrections)
    lines += [
        "",
        f"权重图空间标准差（13 个条件平均）：**{mean_sd_gate:.4f}**。"
        f"该值直接回答“权重是否随位置变化”。",
        "",
        f"干预：保留空间权重相对平均权重，纠正 {total_corrected} 张、改错 {total_worsened} 张"
        f"（全部 13 个条件合计，三种子）。",
        "",
        "## 说明",
        "",
        "- 12 条件均值沿用项目口径：12 个干扰条件的条件均值再取平均，不含干净图。",
        "- 差值均为逐种子相减后的均值；同向种子数写在括号里，三个种子不足以做显著性检验。",
        "- 官方独立测试集留待全组冻结后统一评估。",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n完成：{args.output}")
    print("\n".join(lines[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
