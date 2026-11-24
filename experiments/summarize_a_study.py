"""
用途：把 A 的十三条件开发集评估结果汇总为报告表格、配对差值与强度曲线。
核心职责：
- 校验评估目录的完整性：每个配置三种子、十三条件、逐图预测行数与指标一致。
- 输出逐运行、三种子汇总、配对差值、干扰族汇总、逐类 F1 和成本表。
- 生成强度曲线与条件柱状图，并写出中文汇总草稿，供 D 直接引用。
关键副作用/安全约束：
- 只读评估目录，只写指定的报告目录；拒绝覆盖已存在的报告目录内容。
- 不重新训练、不挑选最好种子或条件；所有条件都会出现在表内。
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from a_conditions import LEVELS, SEVERITIES

CONDITION_ORDER = [("clean", "none")]
CONDITION_ORDER += [(kind, severity) for kind in LEVELS for severity in SEVERITIES]
FAMILIES = tuple(LEVELS)
AUGMENTATION_ORDER = ("repeat_clean", "targeted_motion", "mixed")


def _read_csv(path, required):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"Missing columns in {path}: {required}")
        return list(reader)


def _number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric {name}: {value!r}") from error
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {name}: {value!r}")
    return result


def _mean_sd(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Empty aggregate")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def _config_label(input_name, augmentation, single_input):
    return augmentation if single_input else f"{augmentation}/{input_name}"


def load_eval(directory):
    """Read one `evaluate_a13.py` output tree and check it against its completion record."""
    directory = Path(directory)
    if (directory / "debug_limited.json").is_file():
        raise ValueError(f"{directory} is a limited debug run; rerun the full evaluation")
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    if completion.get("limited"):
        raise ValueError("Completion record marks this as a limited debug run")
    if completion.get("conditions") != len(CONDITION_ORDER):
        raise ValueError("Completion record does not describe thirteen conditions")
    runs = []
    for record in completion["runs"]:
        run_id = record["run_id"]
        run_dir = directory / run_id
        metric_rows = _read_csv(run_dir / "metrics.csv", ("run_id", "seed", "augmentation", "input",
                                                          "condition", "severity", "macro_f1",
                                                          "accuracy", "n_images"))
        prediction_rows = _read_csv(run_dir / "predictions.csv",
                                    ("run_id", "split", "image", "condition", "severity",
                                     "y_true", "y_pred"))
        cost_rows = _read_csv(run_dir / "costs.csv", ("condition", "severity",
                                                      "preprocessing_seconds", "evaluation_seconds"))
        metrics, sizes = {}, {}
        for row in metric_rows:
            key = (row["condition"], row["severity"])
            if key in metrics:
                raise ValueError(f"Duplicate condition in {run_dir}: {key}")
            metrics[key] = {
                "macro_f1": _number(row["macro_f1"], "macro_f1"),
                "accuracy": _number(row["accuracy"], "accuracy"),
            }
            sizes[key] = int(row["n_images"])
        if set(metrics) != set(CONDITION_ORDER):
            raise ValueError(f"{run_dir} does not contain the thirteen frozen conditions")
        if len(set(sizes.values())) != 1:
            raise ValueError(f"{run_dir} evaluates different image counts per condition")
        predictions = {}
        for row in prediction_rows:
            predictions.setdefault((row["condition"], row["severity"]), []).append(
                (int(row["y_true"]), int(row["y_pred"])))
        for key, count in sizes.items():
            if len(predictions.get(key, [])) != count:
                raise ValueError(f"Prediction rows do not match metrics for {run_id} {key}")
        costs = {(row["condition"], row["severity"]): {
            "preprocessing_seconds": _number(row["preprocessing_seconds"], "preprocessing_seconds"),
            "evaluation_seconds": _number(row["evaluation_seconds"], "evaluation_seconds"),
        } for row in cost_rows}
        runs.append({
            "run_id": run_id, "seed": int(metric_rows[0]["seed"]),
            "augmentation": metric_rows[0]["augmentation"], "input": metric_rows[0]["input"],
            "metrics": metrics, "predictions": predictions, "costs": costs,
            "n_images": sizes[CONDITION_ORDER[0]],
        })
    seeds = {}
    for run in runs:
        seeds.setdefault((run["input"], run["augmentation"]), set()).add(run["seed"])
    for key, values in seeds.items():
        if len(values) != 3:
            raise ValueError(f"{key} has {len(values)} seeds, expected three: {sorted(values)}")
    if len(runs) % 3:
        raise ValueError("Run count is not a multiple of three seeds")
    return runs, completion


def by_run(runs):
    rows = []
    for run in sorted(runs, key=lambda item: (item["input"], item["augmentation"], item["seed"])):
        for condition, severity in CONDITION_ORDER:
            value = run["metrics"][(condition, severity)]
            rows.append({
                "run_id": run["run_id"], "input": run["input"],
                "augmentation": run["augmentation"], "seed": run["seed"],
                "condition": condition, "severity": severity,
                "macro_f1": 100 * value["macro_f1"], "accuracy": 100 * value["accuracy"],
                "n_images": run["n_images"],
            })
    return rows


def summarize(runs, single_input):
    groups = {}
    for run in runs:
        groups.setdefault((run["input"], run["augmentation"]), []).append(run)
    rows = []
    for (input_name, augmentation), members in sorted(groups.items()):
        for condition, severity in CONDITION_ORDER:
            f1 = [member["metrics"][(condition, severity)]["macro_f1"] for member in members]
            accuracy = [member["metrics"][(condition, severity)]["accuracy"] for member in members]
            f1_mean, f1_sd = _mean_sd(f1)
            accuracy_mean, accuracy_sd = _mean_sd(accuracy)
            rows.append({
                "config": _config_label(input_name, augmentation, single_input),
                "input": input_name, "augmentation": augmentation,
                "condition": condition, "severity": severity,
                "seed_count": len(members),
                "macro_f1_mean": 100 * f1_mean, "macro_f1_sd": 100 * f1_sd,
                "accuracy_mean": 100 * accuracy_mean, "accuracy_sd": 100 * accuracy_sd,
            })
    return rows


def paired(runs, single_input):
    """Seed-matched differences between augmentation strategies within one input mode."""
    groups = {}
    for run in runs:
        groups.setdefault((run["input"], run["augmentation"]), {})[run["seed"]] = run
    rows = []
    for input_name in sorted({run["input"] for run in runs}):
        present = [aug for aug in AUGMENTATION_ORDER if (input_name, aug) in groups]
        for target_index, target in enumerate(present):
            for baseline in present[:target_index]:
                seeds = sorted(set(groups[(input_name, target)]) & set(groups[(input_name, baseline)]))
                if not seeds:
                    continue
                for condition, severity in CONDITION_ORDER:
                    deltas = [100 * (groups[(input_name, target)][seed]["metrics"]
                                     [(condition, severity)]["macro_f1"]
                                     - groups[(input_name, baseline)][seed]["metrics"]
                                     [(condition, severity)]["macro_f1"])
                              for seed in seeds]
                    mean, sd = _mean_sd(deltas)
                    rows.append({
                        "comparison": f"{_config_label(input_name, target, single_input)}"
                                      f" - {_config_label(input_name, baseline, single_input)}",
                        "input": input_name, "target": target, "baseline": baseline,
                        "condition": condition, "severity": severity, "seeds": len(seeds),
                        "delta_mean": mean, "delta_sd": sd,
                        "seed_deltas": ";".join(f"{value:.4f}" for value in deltas),
                        "improved_seeds": sum(1 for value in deltas if value > 0),
                    })
    return rows


def families(summary_rows):
    """Average each interference family over its three strengths, plus 12-condition mean/worst."""
    rows = []
    for config in dict.fromkeys(row["config"] for row in summary_rows):
        members = [row for row in summary_rows if row["config"] == config]
        lookup = {(row["condition"], row["severity"]): row["macro_f1_mean"] for row in members}
        clean = lookup[("clean", "none")]
        for family in FAMILIES:
            values = [lookup[(family, severity)] for severity in SEVERITIES]
            rows.append({
                "config": config, "family": family, "mean_macro_f1": float(np.mean(values)),
                "worst_macro_f1": float(np.min(values)), "drop_vs_clean": clean - float(np.mean(values)),
                "worst_condition": "",
            })
        interference = [lookup[key] for key in CONDITION_ORDER if key[0] != "clean"]
        worst_key = min(CONDITION_ORDER[1:], key=lambda key: lookup[key])
        rows.append({
            "config": config, "family": "all_interference",
            "mean_macro_f1": float(np.mean(interference)),
            "worst_macro_f1": float(np.min(interference)),
            "drop_vs_clean": clean - float(np.mean(interference)),
            "worst_condition": f"{worst_key[0]}/{worst_key[1]}",
        })
    return rows


def per_class(runs, single_input):
    from sklearn.metrics import f1_score

    classes = sorted({int(label) for run in runs for pairs in run["predictions"].values()
                      for label, _ in pairs})
    groups = {}
    for run in runs:
        groups.setdefault((run["input"], run["augmentation"]), []).append(run)
    rows = []
    for (input_name, augmentation), members in sorted(groups.items()):
        for condition, severity in CONDITION_ORDER:
            collected = {class_id: [] for class_id in classes}
            for run in members:
                pairs = run["predictions"][(condition, severity)]
                scores = f1_score([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                                  labels=classes, average=None, zero_division=0)
                for class_id, score in zip(classes, scores):
                    collected[class_id].append(100 * float(score))
            for class_id in classes:
                mean, sd = _mean_sd(collected[class_id])
                rows.append({
                    "config": _config_label(input_name, augmentation, single_input),
                    "condition": condition, "severity": severity, "class_id": class_id,
                    "f1_mean": mean, "f1_sd": sd,
                })
    return rows


def costs(runs, single_input):
    groups = {}
    for run in runs:
        groups.setdefault((run["input"], run["augmentation"]), []).append(run)
    rows = []
    for (input_name, augmentation), members in sorted(groups.items()):
        for condition, severity in CONDITION_ORDER:
            preprocess = [member["costs"][(condition, severity)]["preprocessing_seconds"]
                          for member in members]
            inference = [member["costs"][(condition, severity)]["evaluation_seconds"]
                         for member in members]
            pre_mean, pre_sd = _mean_sd(preprocess)
            inf_mean, inf_sd = _mean_sd(inference)
            rows.append({
                "config": _config_label(input_name, augmentation, single_input),
                "condition": condition, "severity": severity,
                "images": members[0]["n_images"],
                "preprocessing_seconds_mean": pre_mean, "preprocessing_seconds_sd": pre_sd,
                "evaluation_seconds_mean": inf_mean, "evaluation_seconds_sd": inf_sd,
            })
    return rows


def make_figures(directory, summary_rows, family_rows):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    configs = list(dict.fromkeys(row["config"] for row in summary_rows))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(configs)))
    written = []

    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
    for axis, family in zip(axes.ravel(), FAMILIES):
        for color, config in zip(colors, configs):
            values = [next(row["macro_f1_mean"] for row in summary_rows
                           if row["config"] == config and row["condition"] == family
                           and row["severity"] == severity) for severity in SEVERITIES]
            axis.plot(range(len(SEVERITIES)), values, marker="o", label=config, color=color)
        axis.set_title(family)
        axis.set_xticks(range(len(SEVERITIES)))
        axis.set_xticklabels(SEVERITIES)
        axis.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Macro-F1 (%)")
    axes[1, 0].set_ylabel("Macro-F1 (%)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Development set severity curves (3-seed mean)")
    figure.tight_layout()
    path = directory / "dev13_curves.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(path.name)

    figure, axis = plt.subplots(figsize=(11, 4.5))
    labels = [f"{condition}\n{severity}" for condition, severity in CONDITION_ORDER]
    width = 0.8 / len(configs)
    positions = np.arange(len(CONDITION_ORDER))
    for index, (color, config) in enumerate(zip(colors, configs)):
        values = [next(row["macro_f1_mean"] for row in summary_rows
                       if row["config"] == config and row["condition"] == condition
                       and row["severity"] == severity)
                  for condition, severity in CONDITION_ORDER]
        axis.bar(positions + index * width, values, width, label=config, color=color)
    axis.set_xticks(positions + width * (len(configs) - 1) / 2)
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylabel("Macro-F1 (%)")
    axis.set_title("Development set: all thirteen frozen conditions")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    path = directory / "dev13_conditions.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    written.append(path.name)

    family_path = None
    if family_rows:
        figure, axis = plt.subplots(figsize=(9, 4))
        family_names = [family for family in FAMILIES] + ["all_interference"]
        width = 0.8 / len(configs)
        positions = np.arange(len(family_names))
        for index, (color, config) in enumerate(zip(colors, configs)):
            values = [next(row["mean_macro_f1"] for row in family_rows
                           if row["config"] == config and row["family"] == family)
                      for family in family_names]
            axis.bar(positions + index * width, values, width, label=config, color=color)
        axis.set_xticks(positions + width * (len(configs) - 1) / 2)
        axis.set_xticklabels(family_names, fontsize=8)
        axis.set_ylabel("Macro-F1 (%)")
        axis.set_title("Development set: interference family averages")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
        figure.tight_layout()
        family_path = directory / "dev13_families.png"
        figure.savefig(family_path, dpi=160)
        plt.close(figure)
        written.append(family_path.name)
    return written


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _format(mean, sd):
    return f"{mean:.2f} ± {sd:.2f}"


def build_markdown(completion, summary_rows, paired_rows, family_rows):
    configs = list(dict.fromkeys(row["config"] for row in summary_rows))
    lines = [
        "# A 十三条件开发集汇总",
        "",
        f"数据来源：`{completion['study']}`，{len(completion['runs'])} 次运行，"
        f"每个运行 {completion['images']} 张开发集图像、13 个冻结条件。",
        "",
        "表内为三个训练种子的均值 ± 样本标准差（`ddof=1`），单位为 Macro-F1 百分比。"
        "标准差只描述三个种子的训练随机性，不能当作置信区间。",
        "",
        "## 全部十三条件",
        "",
    ]
    header = ["条件", "强度"] + configs
    rows = []
    for condition, severity in CONDITION_ORDER:
        cells = [condition, severity]
        for config in configs:
            row = next(item for item in summary_rows
                       if item["config"] == config and item["condition"] == condition
                       and item["severity"] == severity)
            cells.append(_format(row["macro_f1_mean"], row["macro_f1_sd"]))
        rows.append(cells)
    lines += [_table(header, rows), ""]

    lines += ["## 干扰族平均与 12 干扰条件", "",
              "每族取三档强度的均值；`all_interference` 为 12 个干扰条件的均值，"
              "`worst` 为其中最差条件。", ""]
    header = ["配置", "族", "均值 Macro-F1", "最差", "相对干净图下降"]
    rows = [[row["config"], row["family"], f"{row['mean_macro_f1']:.2f}",
             f"{row['worst_macro_f1']:.2f}", f"{row['drop_vs_clean']:+.2f}"]
            for row in family_rows]
    lines += [_table(header, rows), ""]

    lines += ["## 配对差值（同种子）", "",
              "差值 = 目标配置 − 基线配置，逐种子相减后再求均值 ± 样本标准差；"
              "`同向种子` 是差值为正的种子数，用于判断稳定性，不能单独当作显著性检验。", ""]
    header = ["比较", "条件", "强度", "差值均值", "差值 SD", "同向种子"]
    rows = [[row["comparison"], row["condition"], row["severity"],
             f"{row['delta_mean']:+.2f}", f"{row['delta_sd']:.2f}",
             f"{row['improved_seeds']}/{row['seeds']}"]
            for row in paired_rows if row["condition"] == "clean"
            or row["severity"] in ("medium", "none")]
    lines += [_table(header, rows), ""]
    lines += ["完整十三条件的配对差值见 `tables/dev13_paired.csv`。", ""]
    return "\n".join(lines)


def write_csv(path, rows, fields=None):
    path = Path(path)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True, help="evaluate_a13.py 的输出目录")
    parser.add_argument("--output", type=Path, default=Path("report/a_part"),
                        help="报告目录，写入 tables/ 与 figures/")
    args = parser.parse_args(argv)

    runs, completion = load_eval(args.eval)
    single_input = len({run["input"] for run in runs}) == 1
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tables").mkdir(exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)

    run_rows = by_run(runs)
    summary_rows = summarize(runs, single_input)
    paired_rows = paired(runs, single_input)
    family_rows = families(summary_rows)
    class_rows = per_class(runs, single_input)
    cost_rows = costs(runs, single_input)

    write_csv(args.output / "tables" / "dev13_by_run.csv", run_rows)
    write_csv(args.output / "tables" / "dev13_summary.csv", summary_rows)
    write_csv(args.output / "tables" / "dev13_paired.csv", paired_rows)
    write_csv(args.output / "tables" / "dev13_families.csv", family_rows)
    write_csv(args.output / "tables" / "dev13_per_class.csv", class_rows)
    write_csv(args.output / "tables" / "dev13_costs.csv", cost_rows)
    figures = make_figures(args.output / "figures", summary_rows, family_rows)
    (args.output / "dev13_summary.md").write_text(
        build_markdown(completion, summary_rows, paired_rows, family_rows), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "eval_directory": Path(args.eval).as_posix(),
        "runs": len(runs),
        "conditions": len(CONDITION_ORDER),
        "images_per_condition": runs[0]["n_images"],
        "seeds": sorted({run["seed"] for run in runs}),
        "configs": list(dict.fromkeys(row["config"] for row in summary_rows)),
        "tables": ["dev13_by_run.csv", "dev13_summary.csv", "dev13_paired.csv",
                   "dev13_families.csv", "dev13_per_class.csv", "dev13_costs.csv"],
        "figures": figures,
        "completion_sha256": hashlib.sha256(
            (Path(args.eval) / "completion.json").read_bytes()).hexdigest(),
    }
    (args.output / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summarized {len(runs)} runs x {len(CONDITION_ORDER)} conditions into {args.output}")
    print(f"Tables: {len(manifest['tables'])}, figures: {figures}")


if __name__ == "__main__":
    main()
