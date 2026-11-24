"""
用途：核对两个成员是否真的用了同一批开发集输入（条件指纹与图像清单）。
核心职责：
- 比较 images.csv 的图像集合、顺序、标签与轨迹。
- 比较 conditions.json 的划分指纹、条件参数和每个条件的像素 SHA256。
- 也接受训练研究目录，比较 run_config.json 与 data_manifest.json 的数据指纹。
关键副作用/安全约束：
- 只读两个目录，不生成图像、不加载模型、不需要 OpenCV 或 GPU。
"""

import argparse
import csv
import json
from pathlib import Path
import sys


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_images(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [(row["image"], int(row["class_id"]), row.get("track_id", ""), row.get("split", ""))
            for row in rows]


def load_conditions(directory):
    """Accept an evaluation directory or a study directory and normalise what we can compare."""
    directory = Path(directory)
    conditions_path = directory / "conditions.json"
    images_path = directory / "images.csv"
    if conditions_path.is_file():
        payload = read_json(conditions_path)
        return {
            "kind": "evaluation",
            "conditions": payload,
            "images": read_images(images_path) if images_path.is_file() else None,
            "data_fingerprint": payload.get("data_fingerprint"),
            "split_sha256": payload.get("split_sha256"),
            "levels": payload.get("levels"),
            "noise_seed": payload.get("noise_seed"),
            "entries": payload.get("conditions", []),
        }
    run_config = directory / "run_config.json"
    if run_config.is_file():
        common = read_json(run_config)["common"]
        manifest = read_json(directory / "data_manifest.json")
        return {
            "kind": "study",
            "conditions": None,
            "images": None,
            "data_fingerprint": common.get("data_fingerprint"),
            "split_sha256": common.get("split_sha256"),
            "levels": None,
            "noise_seed": None,
            "entries": [],
            "generated": manifest.get("generated_fingerprints", {}),
        }
    raise SystemExit(f"{directory} 既没有 conditions.json 也没有 run_config.json")


def compare(reference, candidate):
    problems = []
    report = {}

    if reference["split_sha256"] != candidate["split_sha256"]:
        problems.append("划分文件指纹不同：两边的 pilot_split.csv 不是同一份")
        report["split_sha256"] = {"reference": reference["split_sha256"],
                                  "candidate": candidate["split_sha256"], "match": False}
    else:
        report["split_sha256"] = {"match": True}

    if reference["images"] is not None and candidate["images"] is not None:
        same_order = reference["images"] == candidate["images"]
        same_set = set(reference["images"]) == set(candidate["images"])
        report["images"] = {
            "reference": len(reference["images"]),
            "candidate": len(candidate["images"]),
            "same_order": same_order,
            "same_set": same_set,
        }
        if not same_set:
            missing = sorted(set(reference["images"]) - set(candidate["images"]))[:5]
            extra = sorted(set(candidate["images"]) - set(reference["images"]))[:5]
            problems.append(f"图像清单不同：缺 {len(set(reference['images']) - set(candidate['images']))} 张，"
                            f"多 {len(set(candidate['images']) - set(reference['images']))} 张")
            report["images"]["missing_examples"] = missing
            report["images"]["extra_examples"] = extra
        elif not same_order:
            problems.append("图像集合相同但顺序不同：逐图对齐会错位")

    if reference["levels"] != candidate["levels"] and candidate["levels"] is not None:
        problems.append("条件参数表不同")
    if reference["noise_seed"] != candidate["noise_seed"] and candidate["noise_seed"] is not None:
        problems.append("噪声种子不同")

    if reference["data_fingerprint"] and candidate["data_fingerprint"]:
        match = reference["data_fingerprint"] == candidate["data_fingerprint"]
        report["data_fingerprint"] = {"match": match,
                                      "reference": reference["data_fingerprint"],
                                      "candidate": candidate["data_fingerprint"]}
        if not match:
            problems.append("整体 data_fingerprint 不同（生成像素不一致）")

    if reference["entries"] and candidate["entries"]:
        candidate_index = {(entry["condition"], entry["severity"]): entry
                           for entry in candidate["entries"]}
        table, mismatches = [], 0
        for entry in reference["entries"]:
            key = (entry["condition"], entry["severity"])
            other = candidate_index.get(key)
            same = other is not None and other.get("pixel_sha256") == entry.get("pixel_sha256")
            if not same:
                mismatches += 1
            table.append({
                "condition": key[0], "severity": key[1],
                "parameter": entry.get("parameter"),
                "match": same,
                "reference": (entry.get("pixel_sha256") or "")[:12],
                "candidate": ((other or {}).get("pixel_sha256") or "-")[:12],
            })
        report["conditions"] = {"compared": len(table), "same": len(table) - mismatches,
                                "different": mismatches, "table": table}
        if mismatches:
            problems.append(f"13 个条件里有 {mismatches} 个像素不一致")
    elif reference["entries"] and candidate["kind"] == "study":
        problems.append("对方只提供了训练目录，没有开发集 conditions.json，无法逐条件比对")

    return report, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="基准目录，例如 results/a_eval13_local_v1")
    parser.add_argument("--candidate", type=Path, required=True, help="待核对的另一个成员的目录")
    parser.add_argument("--json", type=Path, help="把完整结果写到这个文件")
    args = parser.parse_args(argv)

    reference = load_conditions(args.reference)
    candidate = load_conditions(args.candidate)
    report, problems = compare(reference, candidate)

    print(f"基准：{args.reference}（{reference['kind']}）")
    print(f"对照：{args.candidate}（{candidate['kind']}）")
    print()
    print("划分指纹：" + ("一致" if report.get("split_sha256", {}).get("match") else "不一致"))
    if "images" in report:
        info = report["images"]
        print(f"图像清单：{info['reference']} vs {info['candidate']} 张，"
              f"集合{'一致' if info['same_set'] else '不同'}，"
              f"顺序{'一致' if info['same_order'] else '不同'}")
    if "conditions" in report:
        info = report["conditions"]
        print(f"条件像素：{info['same']}/{info['compared']} 一致")
        print()
        print(f"{'条件':<10}{'强度':<8}{'参数':<8}{'结果':<6}{'基准':<14}{'对照':<14}")
        for row in info["table"]:
            print(f"{row['condition']:<10}{row['severity']:<8}{str(row['parameter']):<8}"
                  f"{'一致' if row['match'] else '不同':<6}{row['reference']:<14}{row['candidate']:<14}")
    print()
    if problems:
        print("结论：**不能直接配对**")
        for item in problems:
            print("  - " + item)
    else:
        print("结论：输入一致，可以配对比较。")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"reference": str(args.reference), "candidate": str(args.candidate),
                        "report": report, "problems": problems}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"\n完整结果：{args.json}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
