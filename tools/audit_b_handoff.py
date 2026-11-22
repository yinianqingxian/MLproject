"""
用途：只用标准库验收 B 的增强视图与单分支交接资料并汇总其内部对照。
核心职责：
- 核对冻结划分、十二次配置、十三条件指标、计时和来源指纹。
- 导出三种子统计与去噪配对，区分资料结构检查和完整训练验收。
关键副作用/安全约束：
- 只读取来源，不导入交接代码、不训练、不生成 C 成绩；输出目录必须新建。
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


PROJECT = Path(__file__).resolve().parents[1]
CODE = ("run_resnet_study.py", "resnet_robustness_study.py", "run_c_study.py",
        "cnn_robustness_study.py", "cnn_robustness_pilot.py")
WEIGHTS_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
LEVELS = {"motion": [3, 5, 7], "noise": [10, 20, 30],
          "dark": [0.7, 0.45, 0.25], "jpeg": [40, 20, 10]}
CONDITIONS = {("clean", "none"): None, **{
    (kind, severity): value for kind, values in LEVELS.items()
    for severity, value in zip(("low", "medium", "high"), values)}}
PAIRS = (("repeat_clean", "clahe"), ("targeted_motion", "clahe"),
         ("mixed", "clahe"), ("mixed", "denoised_clahe"))
RUNS = {f"{a}_{m}_seed{s}": (a, m, s) for a, m in PAIRS for s in (42, 43, 44)}
FIXED = {"backbone": "torchvision.models.resnet18", "feature_dim": 512,
         "network_input_size": 224, "image_size": 32, "classes": [1, 2, 3, 4, 5, 7, 8],
         "epochs": 10, "batch_size": 128, "learning_rate": 0.003, "optimizer": "AdamW",
         "weight_decay": 0.01, "training_entries": 4410, "clean_training_images": 2940,
         "validation_images": 840, "train_tracks": 98, "validation_tracks": 28,
         "clahe": {"clip_limit": 2.0, "tile_grid": [4, 4], "channel": "LAB_L"},
         "denoising": {"method": "bilateral", "d": 5, "sigmaColor": 25, "sigmaSpace": 2},
         "pretrained": True, "weights_id": "ResNet18_Weights.IMAGENET1K_V1",
         "weights_sha256": WEIGHTS_SHA}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        require(len(headers) == len(set(headers)), f"Duplicate CSV columns: {path}")
        require(set(required) <= set(headers), f"Missing CSV columns: {path}")
        rows = list(reader)
        require(all(None not in row and all(value is not None for value in row.values())
                    for row in rows), f"Invalid CSV row width: {path}")
        return rows


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def number(value, label, lower=0, upper=None):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}: {value}") from error
    require(math.isfinite(value) and value >= lower and (upper is None or value <= upper),
            f"Invalid {label}: {value}")
    return value


def verify_source_manifest(source):
    files = {p.relative_to(source).as_posix(): p for p in source.rglob("*") if p.is_file()}
    require(not any(p.is_symlink() for p in source.rglob("*")), "Source manifest: symbolic links refused")
    if "SOURCE_MANIFEST.json" not in files:
        return {"present": False, "verified_files": 0}
    entries = read_json(source / "SOURCE_MANIFEST.json").get("files", [])
    paths = [entry.get("path") for entry in entries]
    require(len(paths) == len(set(paths)), "Source manifest duplicate paths")
    require(set(paths) == set(files) - {"SOURCE_MANIFEST.json"}, "Source manifest inventory mismatch")
    for entry in entries:
        name = entry["path"]
        require(isinstance(name, str) and "\\" not in name and ":" not in name
                and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                "Source manifest unsafe path")
        path = files[name]
        require(path.stat().st_size == entry.get("bytes") and digest(path) == entry.get("sha256"),
                f"Source manifest fingerprint mismatch: {name}")
    return {"present": True, "verified_files": len(entries)}


def keyed_conditions(rows, label):
    mapping = {}
    for row in rows:
        key = row["condition"], row["severity"]
        require(key not in mapping, f"Duplicate {label} condition: {key}")
        mapping[key] = row
    require(set(mapping) == set(CONDITIONS), f"Incomplete {label} conditions")
    return mapping


def audit_handoff(source: Path, project: Path) -> dict:
    """Validate metadata and summarize B only; no imported scripts or model loading."""
    source, project = Path(source).resolve(), Path(project).resolve()
    source_manifest = verify_source_manifest(source)
    conditions = read_json(source / "清单/conditions.json")
    require(conditions.get("levels") == LEVELS and conditions.get("noise_seed") == 20261117,
            "Unexpected thirteen-condition parameters")
    condition_rows = keyed_conditions(conditions["conditions"], "manifest")
    for key, row in condition_rows.items():
        require(row.get("parameter") == CONDITIONS[key] and row.get("n_images") == 840
                and is_sha(row.get("pixel_sha256")), f"Invalid condition manifest: {key}")
    require(is_sha(conditions.get("data_fingerprint")), "Invalid data_fingerprint")
    split_path = source / "清单/pilot_split.csv"
    split_sha = digest(split_path)
    require(split_sha == conditions.get("split_sha256") == digest(project / "results/pilot_split.csv"),
            "Frozen split SHA mismatch")
    fields = ("image", "class_id", "track_id", "split")
    split = read_csv(split_path, fields)
    train = [r for r in split if r["split"] == "train"]
    validation = [r for r in split if r["split"] == "validation"]
    require(len(split) == 3780 and len({r["image"] for r in split}) == 3780
            and (len(train), len(validation)) == (2940, 840), "Invalid split image budget")
    train_tracks, valid_tracks = ({r["track_id"] for r in rows} for rows in (train, validation))
    require((len(train_tracks), len(valid_tracks)) == (98, 28) and not train_tracks & valid_tracks,
            "Split trajectory overlap or invalid track budget")
    images = read_csv(source / "清单/images.csv", fields)
    require(len(images) == 840 and len({r["image"] for r in images}) == 840
            and {tuple(r[k] for k in fields) for r in images} ==
            {tuple(r[k] for k in fields) for r in validation}, "Validation images/track manifest mismatch")
    config_paths = {p.parent.name: p for p in (source / "配置").glob("*/config.json")}
    require(set(config_paths) == set(RUNS), "Incomplete or unexpected twelve-configuration matrix")
    configs = {run: read_json(path) for run, path in config_paths.items()}
    for run, config in configs.items():
        require((config.get("augmentation"), config.get("input"), config.get("seed")) == RUNS[run]
                and config.get("run_id") == run, f"Configuration identity mismatch: {run}")
        for key, expected in FIXED.items():
            require(config.get(key) == expected and (key != "pretrained" or config[key] is True),
                    f"Configuration {run}: invalid {key}")
        require(config.get("data_fingerprint") == conditions["data_fingerprint"]
                and config.get("split_sha256") == split_sha, f"Configuration data fingerprint: {run}")
        code = config.get("code_fingerprints", {})
        require(set(code) == set(CODE) and all(is_sha(v) for v in code.values()),
                f"Invalid source declarations: {run}")
    first = next(iter(configs.values()))
    for run, config in configs.items():
        require(config["code_fingerprints"] == first["code_fingerprints"], f"Source declarations differ: {run}")
    metrics = read_csv(source / "B_13条件指标汇总.csv",
                       ("run_id", "seed", "augmentation", "input", "split", "condition", "severity",
                        "macro_f1", "accuracy", "n_images", "parameters"))
    expected_keys = {(run, *key) for run in RUNS for key in CONDITIONS}
    indexed = {}
    for row in metrics:
        key = row["run_id"], row["condition"], row["severity"]
        require(key in expected_keys and key not in indexed, f"Unexpected/duplicate metric key: {key}")
        require((row["augmentation"], row["input"], int(row["seed"])) == RUNS[row["run_id"]]
                and row["split"] == "validation" and int(row["n_images"]) == 840
                and int(row["parameters"]) == 11180103, f"Invalid metric metadata: {key}")
        for field in ("macro_f1", "accuracy"):
            row[field] = number(row[field], f"metric {field}", upper=1)
        indexed[key] = row
    require(set(indexed) == expected_keys, "Incomplete metric matrix")
    cost_paths = {p.parent.name: p for p in (source / "计时").glob("*/costs.csv")}
    require(set(cost_paths) == set(RUNS), "Incomplete cost run matrix")
    for run, path in cost_paths.items():
        costs = keyed_conditions(read_csv(path, ("condition", "severity", "preprocessing_seconds",
                                                "evaluation_seconds", "images")), "cost")
        for key, row in costs.items():
            require(int(row["images"]) == 840, f"Invalid cost image budget: {run}/{key}")
            for field in ("preprocessing_seconds", "evaluation_seconds"):
                number(row[field], f"cost {field}")
    mac = read_json(source / "成员B样例/输入核对_Mac.json")
    mac_rows = keyed_conditions(mac["conditions"], "Mac input audit")
    matching, mismatching = [], []
    for key, row in mac_rows.items():
        require(row.get("count") == 840 and row.get("local_input") == condition_rows[key]["pixel_sha256"]
                and is_sha(row.get("a_reference_input")), f"Mac input audit inconsistent: {key}")
        matches = row["local_input"] == row["a_reference_input"]
        require(row.get("matches_a") is matches, f"Mac input audit match flag inconsistent: {key}")
        (matching if matches else mismatching).append({"condition": key[0], "severity": key[1]})
    summary, pairs, paired_summary = [], [], []
    for augmentation, mode in PAIRS:
        for condition, severity in CONDITIONS:
            group = [indexed[f"{augmentation}_{mode}_seed{s}", condition, severity] for s in (42, 43, 44)]
            row = {"augmentation": augmentation, "input": mode, "condition": condition,
                   "severity": severity, "n_seeds": 3}
            for field in ("macro_f1", "accuracy"):
                row[field + "_mean"] = statistics.mean(r[field] for r in group)
                row[field + "_sd"] = statistics.stdev(r[field] for r in group)
            summary.append(row)
    for condition, severity in CONDITIONS:
        group = []
        for seed in (42, 43, 44):
            plain = indexed[f"mixed_clahe_seed{seed}", condition, severity]
            denoised = indexed[f"mixed_denoised_clahe_seed{seed}", condition, severity]
            row = {"condition": condition, "severity": severity, "seed": seed}
            for field in ("macro_f1", "accuracy"):
                row["clahe_" + field], row["denoised_" + field] = plain[field], denoised[field]
                row[field + "_delta"] = denoised[field] - plain[field]
            group.append(row)
        pairs.extend(group)
        row = {"condition": condition, "severity": severity, "n_pairs": 3}
        for field in ("macro_f1", "accuracy"):
            row[field + "_delta_mean"] = statistics.mean(r[field + "_delta"] for r in group)
            row[field + "_delta_sd"] = statistics.stdev(r[field + "_delta"] for r in group)
        paired_summary.append(row)
    current_sources = {name: digest(project / "experiments" / name) for name in CODE}
    weights = project / "assets/weights/resnet18-f37072fd.pth"
    artifact_names = ("model.pt", "predictions.csv", "training_log.csv", "completion.json")
    artifacts = {name: len(list(source.rglob(name))) for name in artifact_names}
    artifacts["feature_archives"] = len(list(source.rglob("*.npz")))
    return {"status": "handoff_structure_verified_with_limitations", "source": str(source),
            "counts": {"runs": 12, "metric_rows": len(metrics), "cost_rows": 156, "images": 840,
                       "conditions": 13}, "source_manifest": source_manifest,
            "input_reference": {"matching_conditions": len(matching), "mismatching_conditions": len(mismatching),
                                "matches": matching, "mismatches": mismatching},
            "source_declarations": {name: {"declared_sha256": first["code_fingerprints"][name],
                                           "current_sha256": current_sources[name],
                                           "matches": first["code_fingerprints"][name] == current_sources[name]}
                                    for name in CODE},
            "official_initialization": {"declared_sha256": WEIGHTS_SHA, "current_file_present": weights.is_file(),
                                        "current_file_matches": weights.is_file() and digest(weights) == WEIGHTS_SHA},
            "reported_environment": first.get("environment", {}), "data_fingerprint": conditions["data_fingerprint"],
            "training_artifact_counts": artifacts, "training_fully_verified": False,
            "safe_for_cross_member_paired_comparison": False, "standard_deviation_ddof": 1,
            "limitations": ["Metadata and B-internal summaries only; metrics cannot be independently recomputed without predictions.",
                            "Model reload, completed-training integrity and epoch budget are not verified by this handoff audit.",
                            "Cross-member condition and view identities must match before paired comparisons; no C scores were generated.",
                            "Different data fingerprints do not establish which training pixels differ.",
                            "Mac CPU timing and Spark GPU timing are not a same-device fusion cost comparison."],
            "tables": {"b_metric_summary": summary, "b_mixed_denoising_pairs": pairs,
                       "b_mixed_denoising_summary": paired_summary}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=PROJECT)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output exists; use a new directory: {args.output}")
    require(not args.output.resolve().is_relative_to(args.source.resolve()),
            "Output must not overlap the read-only handoff source")
    result = audit_handoff(args.source, args.project)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "audit.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    for name, rows in result["tables"].items():
        with (args.output / (name + ".csv")).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Verified B handoff structure: 12 runs, 156 metrics; output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
