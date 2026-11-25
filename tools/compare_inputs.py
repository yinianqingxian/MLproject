"""
用途：核对两个成员是否真的用了同一批输入——既比评估条件，也比训练像素。
核心职责：
- 比较 images.csv 的图像集合、顺序、标签与轨迹。
- 比较 conditions.json 的划分指纹、条件参数和每个条件的像素 SHA256。
- 比较 data_manifest.json 的源图像哈希、增强索引、增强数量和各 split 的生成指纹。
- 同一目录里同时有评估清单和训练清单时，一次性比完两项。
关键副作用/安全约束：
- 只读两个目录，不生成图像、不加载模型、不需要 OpenCV、NumPy 或 GPU。
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


def load(directory):
    """Read whatever an evaluation directory and/or a study directory provides."""
    directory = Path(directory)
    result = {
        "kind": [], "images": None, "levels": None, "noise_seed": None, "entries": [],
        "data_fingerprint": None, "split_sha256": None,
        "generated": None, "extra_indices": None, "augmentation_counts": None,
        "source_files": None,
    }

    conditions_path = directory / "conditions.json"
    if conditions_path.is_file():
        payload = read_json(conditions_path)
        result["kind"].append("evaluation")
        result["levels"] = payload.get("levels")
        result["noise_seed"] = payload.get("noise_seed")
        result["entries"] = payload.get("conditions", [])
        result["data_fingerprint"] = payload.get("data_fingerprint")
        result["split_sha256"] = payload.get("split_sha256")
        images_path = directory / "images.csv"
        if images_path.is_file():
            result["images"] = read_images(images_path)

    run_config = directory / "run_config.json"
    if run_config.is_file():
        common = read_json(run_config)["common"]
        result["kind"].append("study")
        result["data_fingerprint"] = result["data_fingerprint"] or common.get("data_fingerprint")
        result["split_sha256"] = result["split_sha256"] or common.get("split_sha256")

    manifest_path = directory / "data_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        result["kind"].append("manifest")
        result["generated"] = manifest.get("generated_fingerprints")
        result["extra_indices"] = manifest.get("extra_indices")
        result["augmentation_counts"] = manifest.get("augmentation_counts")
        result["source_files"] = manifest.get("source_files")
        result["data_fingerprint"] = result["data_fingerprint"] or manifest.get("data_fingerprint")

    if not result["kind"]:
        raise SystemExit(f"{directory} 里没有 conditions.json / run_config.json / data_manifest.json")
    return result


def compare(reference, candidate):
    problems = []
    report = {}

    def same(name, left, right):
        if left is None or right is None:
            return None
        match = left == right
        report[name] = {"match": match}
        return match

    if same("split_sha256", reference["split_sha256"], candidate["split_sha256"]) is False:
        problems.append("划分文件指纹不同：两边的 pilot_split.csv 不是同一份")

    if reference["images"] is not None and candidate["images"] is not None:
        left, right = reference["images"], candidate["images"]
        same_set = set(left) == set(right)
        report["images"] = {"reference": len(left), "candidate": len(right),
                            "same_set": same_set, "same_order": left == right}
        if not same_set:
            missing = len(set(left) - set(right))
            extra = len(set(right) - set(left))
            problems.append(f"图像清单不同：缺 {missing} 张，多 {extra} 张")
            report["images"]["missing_examples"] = sorted(set(left) - set(right))[:5]
            report["images"]["extra_examples"] = sorted(set(right) - set(left))[:5]
        elif left != right:
            problems.append("图像集合相同但顺序不同：逐图对齐会错位")

    if reference["levels"] is not None and candidate["levels"] is not None:
        if reference["levels"] != candidate["levels"]:
            problems.append("条件参数表不同")
        if reference["noise_seed"] != candidate["noise_seed"]:
            problems.append("噪声种子不同")

    if same("data_fingerprint", reference["data_fingerprint"], candidate["data_fingerprint"]) is False:
        problems.append("整体 data_fingerprint 不同（生成像素不一致）")

    if reference["entries"] and candidate["entries"]:
        index = {(entry["condition"], entry["severity"]): entry for entry in candidate["entries"]}
        table, mismatch = [], 0
        for entry in reference["entries"]:
            key = (entry["condition"], entry["severity"])
            other = index.get(key)
            ok = other is not None and other.get("pixel_sha256") == entry.get("pixel_sha256")
            mismatch += 0 if ok else 1
            table.append({"condition": key[0], "severity": key[1], "parameter": entry.get("parameter"),
                          "match": ok, "reference": (entry.get("pixel_sha256") or "")[:12],
                          "candidate": ((other or {}).get("pixel_sha256") or "-")[:12]})
        report["conditions"] = {"compared": len(table), "same": len(table) - mismatch,
                                "different": mismatch, "table": table}
        if mismatch:
            problems.append(f"13 个条件里有 {mismatch} 个像素不一致")

    if reference["source_files"] and candidate["source_files"]:
        if reference["source_files"] == candidate["source_files"]:
            report["source_files"] = {"match": True, "count": len(reference["source_files"])}
        else:
            report["source_files"] = {"match": False}
            problems.append("源图像 sha256 列表不同：两边的 GTSRB 原始文件不是同一份")
    if same("extra_indices", reference["extra_indices"], candidate["extra_indices"]) is False:
        problems.append("增强样本索引不同（随机序列没有复现）")
    if same("augmentation_counts", reference["augmentation_counts"],
            candidate["augmentation_counts"]) is False:
        problems.append("增强数量分配不同")

    if reference["generated"] and candidate["generated"]:
        rows, mismatch = [], 0
        for key in sorted(set(reference["generated"]) | set(candidate["generated"])):
            left = reference["generated"].get(key)
            right = candidate["generated"].get(key)
            ok = left == right and left is not None
            mismatch += 0 if ok else 1
            rows.append({"split": key, "match": ok,
                         "reference": (left or "-")[:12], "candidate": (right or "-")[:12]})
        report["generated"] = {"compared": len(rows), "different": mismatch, "table": rows}
        if mismatch:
            problems.append(f"训练/验证的 {mismatch} 个 split 生成指纹不同")
    return report, problems


def print_report(reference_dir, candidate_dir, reference, candidate, report, problems):
    print(f"基准：{reference_dir}（{' + '.join(reference['kind'])}）")
    print(f"对照：{candidate_dir}（{' + '.join(candidate['kind'])}）")
    print()

    def line(label, key, text=None):
        if key in report:
            print(f"{label}：{text or ('一致' if report[key]['match'] else '不一致')}")

    line("划分指纹", "split_sha256")
    if "images" in report:
        info = report["images"]
        print(f"图像清单：{info['reference']} vs {info['candidate']} 张，"
              f"集合{'一致' if info['same_set'] else '不同'}，"
              f"顺序{'一致' if info['same_order'] else '不同'}")
    if report.get("source_files"):
        print(f"源图像哈希：{'一致' if report['source_files']['match'] else '不同'}"
              f"（{report['source_files'].get('count', '-')} 条）")
    line("增强索引", "extra_indices")
    line("增强数量", "augmentation_counts")
    line("整体 data_fingerprint", "data_fingerprint")

    if "conditions" in report:
        info = report["conditions"]
        print(f"条件像素：{info['same']}/{info['compared']} 一致")
        if info["different"]:
            print()
            print(f"{'条件':<10}{'强度':<8}{'参数':<8}{'结果':<6}{'基准':<14}{'对照':<14}")
            for row in info["table"]:
                if not row["match"]:
                    print(f"{row['condition']:<10}{row['severity']:<8}{str(row['parameter']):<8}"
                          f"{'不同':<6}{row['reference']:<14}{row['candidate']:<14}")
    if "generated" in report:
        info = report["generated"]
        print()
        print(f"{'split':<26}{'结果':<6}{'基准':<14}{'对照':<14}")
        for row in info["table"]:
            print(f"{row['split']:<26}{'一致' if row['match'] else '不同':<6}"
                  f"{row['reference']:<14}{row['candidate']:<14}")

    print()
    if problems:
        print("结论：**不能配对**")
        for item in problems:
            print("  - " + item)
    else:
        print("结论：输入一致，可以配对。")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="基准目录。评估条件用 results/a_eval13_local_v1，训练像素用 results/a_local_v1")
    parser.add_argument("--candidate", type=Path, required=True, help="待核对的另一个成员的目录")
    parser.add_argument("--json", type=Path, help="把完整结果写到这个文件")
    args = parser.parse_args(argv)

    reference = load(args.reference)
    candidate = load(args.candidate)
    report, problems = compare(reference, candidate)
    print_report(args.reference, args.candidate, reference, candidate, report, problems)

    if args.json:
        Path(args.json).write_text(
            json.dumps({"reference": str(args.reference), "candidate": str(args.candidate),
                        "report": report, "problems": problems}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"\n完整结果：{args.json}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
