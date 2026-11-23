#!/usr/bin/env bash
# 用途：A 在 DGX Spark（ARM64 + GB10）上的一站式入口：容器内完成环境检查、九次正式训练、
#       开发集十三条件评估、官方七类测试与汇总。
# 核心职责：
# - 复用仓库自带的 ARM64 NVIDIA 镜像（spark/Dockerfile），不另建第二套环境。
# - 把 A 的脚本放进同一个受保护容器里运行，结果通过 bind mount 留在宿主机。
# - 训练前先用真实 batch 128 探针确认显存与耗时，并核对数据、权重和冻结划分。
# 关键副作用：首次 setup 会构建/拉取镜像；正式结果写入 results/ 下的独立目录。
set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Usage: bash A_SPARK.sh <command> [extra python arguments...]

Commands:
  env      宿主机检查：架构、Docker、GPU、磁盘、内存、数据与权重是否就位。
  setup    构建/复用 ARM64 镜像，然后立刻跑 check。
  check    容器内：GPU 与 batch-128 显存探针 + 数据预算审计 + 全部回归测试。
  smoke    容器内试跑：mixed × 种子 42 × 1 轮（不计入正式结果）。
  train    容器内正式九次训练：raw × 三种增强 × 种子 42/43/44，10 轮。
  resume   中断后续跑，只跳过已完整通过核验的运行。
  verify   容器内只读核验九个运行与全部产物指纹。
  eval13   容器内开发集十三条件评估（不重新训练）。
  report   容器内汇总表格、配对差值、强度曲线到 report/a_part。
  freeze   写出官方测试评估的冻结清单。
  test7    官方七类测试子集独立评估（需先 freeze）。
  status   宿主机查看各输出目录进度。
  shell    进入同一个容器交互排查。
  all      check → smoke → train → verify → eval13 → report

Paths are relative to the project root, which is mounted at /workspace.
环境变量：A_IMAGE（默认复用 C 的镜像名）、A_DEVICE、A_THREADS、A_OUTPUT、A_SMOKE_OUTPUT、
          A_EVAL_OUTPUT、A_REPORT_DIR、A_TEST_FREEZE、A_TEST_OUTPUT。
默认输出目录：results/a_spark_v1、results/a_spark_smoke_v1、results/a_eval13_spark_v1、
          results/a_test7_v1 —— 与 Windows/Linux 的结果分开，避免混入同一份均值。
USAGE
}

fail() { printf 'A_SPARK: %s\n' "$*" >&2; exit 1; }

command_name="${1:-help}"
[[ "$command_name" == help || "$command_name" == -h || "$command_name" == --help ]] && { usage; exit 0; }
forward=("${@:2}")

project_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_ref="${A_IMAGE:-ds4023-c-resnet18-spark:25.11}"
device="${A_DEVICE:-cuda}"
threads="${A_THREADS:-6}"
output="${A_OUTPUT:-results/a_spark_v1}"
smoke_output="${A_SMOKE_OUTPUT:-results/a_spark_smoke_v1}"
eval_output="${A_EVAL_OUTPUT:-results/a_eval13_spark_v1}"
report_dir="${A_REPORT_DIR:-report/a_part_spark}"
freeze="${A_TEST_FREEZE:-results/a_test7_freeze.json}"
test_output="${A_TEST_OUTPUT:-results/a_test7_v1}"
data_root="${A_DATA_ROOT:-data/GTSRB/raw}"

case "$command_name" in
    env)
        echo "== 架构与系统 =="
        uname -a
        echo
        echo "== GPU =="
        nvidia-smi || echo "WARNING: nvidia-smi 不可用"
        echo
        echo "== Docker =="
        (docker info >/dev/null 2>&1 && echo "docker OK（当前用户可连接守护进程）") \
            || echo "FAIL: docker 不可用或无权限，见 DGX_SPARK.md"
        echo
        echo "== 磁盘与内存 =="
        df -h "$project_root" || true
        free -g || true
        echo
        echo "== 协议文件 =="
        for path in "$data_root/GT-final_test.csv" "$data_root/GTSRB/Final_Training/Images" \
                    "$data_root/GTSRB/Final_Test/Images" "results/pilot_split.csv" \
                    "assets/weights/resnet18-f37072fd.pth"; do
            if [[ -e "$project_root/$path" ]]; then echo "OK      $path"; else echo "MISSING $path"; fi
        done
        echo
        echo "== 现有结果目录 =="
        for dir in "$output" "$smoke_output" "$eval_output" "$test_output"; do
            [[ -e "$project_root/$dir" ]] && echo "EXISTS  $dir" || echo "absent  $dir"
        done
        echo
        echo "== GPU 上是否已有别的进程占用 =="
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
        exit 0
        ;;
    status)
        for dir in "$output" "$smoke_output" "$eval_output" "$test_output"; do
            if [[ -d "$project_root/$dir" ]]; then
                runs=$(find "$project_root/$dir/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
                printf '%-40s %2s runs  %s\n' "$dir" "$runs" "$(du -sh "$project_root/$dir" 2>/dev/null | cut -f1)"
            else
                printf '%-40s (missing)\n' "$dir"
            fi
        done
        exit 0
        ;;
    setup|check|smoke|train|resume|verify|eval13|report|freeze|test7|shell|all) ;;
    *) usage >&2; exit 2 ;;
esac

[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] \
    || fail '实际运行需要 Linux aarch64；这是 DGX Spark 专用入口。'
command -v docker >/dev/null 2>&1 || fail '找不到 docker，见 DGX_SPARK.md 的容器运行时说明。'
docker info >/dev/null 2>&1 || fail 'Docker 守护进程不可用或当前用户无权限，见 DGX_SPARK.md。'

if [[ "$command_name" == setup ]]; then
    printf '==> 构建 ARM64 镜像 %s（复用 spark/Dockerfile）\n' "$image_ref"
    docker build --platform linux/arm64 --tag "$image_ref" "$project_root/spark"
    command_name=check
fi

image_details="$(docker image inspect --format '{{.Architecture}}|{{.Id}}' "$image_ref" 2>/dev/null)" \
    || fail "镜像 $image_ref 不存在。先运行 bash A_SPARK.sh setup。"
image_arch="${image_details%%|*}"
image_id="${image_details#*|}"
[[ "$image_arch" == arm64 ]] || fail "镜像架构是 $image_arch，需要 linux/arm64；请用 setup 重建。"
[[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'Docker 返回了无效的镜像 ID。'

case "$command_name" in
    check)
        payload=(bash -lc "set -e
python tools/check_a_linux.py --weights assets/weights/resnet18-f37072fd.pth --output results/a_spark_environment
python tools/audit_a_data.py --data-root $data_root --output results/a_spark_environment/data_audit
python -m unittest discover -s experiments/tests -p 'test_c_data_protocol.py' -v
python -m unittest discover -s experiments/tests -p 'test_a_conditions.py' -v
python -m unittest discover -s experiments/tests -p 'test_a_evaluate.py' -v
python -m unittest discover -s experiments/tests -p 'test_a_summary.py' -v") ;;
    smoke)
        payload=(python experiments/run_resnet_study.py --inputs raw --augmentations mixed
                 --seeds 42 --epochs 1 --threads "$threads" --device "$device"
                 --data-root "$data_root" --output-dir "$smoke_output") ;;
    train)
        payload=(python experiments/run_resnet_study.py --inputs raw
                 --augmentations repeat_clean targeted_motion mixed --seeds 42 43 44
                 --epochs 10 --threads "$threads" --device "$device" --data-root "$data_root"
                 --output-dir "$output") ;;
    resume)
        payload=(python experiments/run_resnet_study.py --inputs raw
                 --augmentations repeat_clean targeted_motion mixed --seeds 42 43 44
                 --epochs 10 --threads "$threads" --device "$device" --data-root "$data_root"
                 --output-dir "$output" --resume) ;;
    verify)
        payload=(python experiments/run_resnet_study.py --inputs raw
                 --augmentations repeat_clean targeted_motion mixed --seeds 42 43 44
                 --epochs 10 --threads "$threads" --device "$device" --data-root "$data_root"
                 --output-dir "$output" --verify-only) ;;
    eval13)
        payload=(python experiments/evaluate_a13.py --study "$output" --output "$eval_output"
                 --device "$device" --data-root "$data_root" --expect-runs 9) ;;
    report)
        payload=(python experiments/summarize_a_study.py --eval "$eval_output" --output "$report_dir") ;;
    freeze)
        payload=(python experiments/evaluate_official_test.py --init-freeze "$freeze" --study "$output") ;;
    test7)
        payload=(python experiments/evaluate_official_test.py --freeze "$freeze" --output "$test_output"
                 --device "$device" --data-root "$data_root") ;;
    shell)
        payload=(bash) ;;
    all)
        "$0" check
        "$0" smoke
        "$0" train
        "$0" verify
        "$0" eval13
        "$0" report
        exit 0 ;;
esac

run=(docker run --rm --init --platform linux/arm64 --gpus all --shm-size 2g
     --user "$(id -u):$(id -g)" --mount "type=bind,source=$project_root,target=/workspace"
     --workdir /workspace --env HOME=/tmp --env MPLCONFIGDIR=/tmp/mpl-cache --env MPLBACKEND=Agg
     --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1
     --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env NVIDIA_IMEX_CHANNELS=0
     --env NVIDIA_TF32_OVERRIDE=0
     --env "SPARK_CONTAINER_IMAGE_ID=$image_id" --env "SPARK_CONTAINER_IMAGE_REF=$image_ref")
if [[ "$command_name" == shell ]]; then
    run+=(-it)
fi
run+=("$image_id" "${payload[@]}" "${forward[@]}")
printf '==> %s\n' "$command_name"
exec "${run[@]}"
