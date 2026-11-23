#读取划分清单和逐图预测，核对图片、标签、种子及评估条件。
import csv

SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "motion", "noise", "dark", "jpeg")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def load_predictions(results):
    labels = {row["image"]: row["class_id"]
              for row in read_rows(results / "pilot_split.csv")
              if row["split"] == "validation"}
    classes = set(labels.values())
    if len(labels) != 840 or len(classes) != 7:
        raise ValueError("Expected 840 validation images in seven classes.")

    groups = {(f"{model}_seed{seed}", condition): {}
              for model in ("mixed_dual", "mixed_fixed")
              for seed in SEEDS for condition in CONDITIONS}
    path = results / "study_resnet18_rtx4060ti_v1" / "predictions.csv"
    for row in read_rows(path):
        key = (row["run_id"], row["condition"])
        if row["split"] != "validation" or key not in groups:
            continue
        image = row["image"]
        severity = "none" if row["condition"] == "clean" else "medium"
        if (labels.get(image) != row["y_true"] or row["y_pred"] not in classes
                or image in groups[key] or row["severity"] != severity
                or not row["run_id"].endswith("_seed" + row["seed"])):
            raise ValueError(f"Invalid prediction for {key}: {image}")
        groups[key][image] = row["y_pred"]

    for key, predictions in groups.items():
        if predictions.keys() != labels.keys():
            raise ValueError(f"Incomplete validation set for {key}")
    return labels, groups
