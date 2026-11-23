#运行入口。执行分析、打印结果并保存 CSV。
from collections import Counter
from statistics import mean, stdev

from prediction_data import CONDITIONS, SEEDS


def macro_f1(labels, predictions):
    actual = Counter(labels.values())
    predicted = Counter(predictions.values())
    correct = Counter(label for image, label in labels.items()
                      if predictions[image] == label)
    return mean(2 * correct[c] / (actual[c] + predicted[c]) for c in actual)


def compare_models(labels, groups):
    scores = {key: macro_f1(labels, predictions) for key, predictions in groups.items()}
    rows = []
    for condition in CONDITIONS:
        row = {"condition": condition}
        differences = []
        for seed in SEEDS:
            gated = scores[f"mixed_dual_seed{seed}", condition]
            fixed = scores[f"mixed_fixed_seed{seed}", condition]
            delta = 100 * (gated - fixed)
            row[f"seed{seed}_pp"] = delta
            differences.append(delta)
        row["mean_pp"] = mean(differences)
        row["sd_pp"] = stdev(differences)
        row["positive_seeds"] = sum(delta > 0 for delta in differences)
        rows.append(row)
    return rows
