#!/usr/bin/env bash
# 用途：在 DGX Spark 上构建并运行 C 部分的独立 ARM64 GPU 环境。
# 核心职责：提供环境检查、冒烟训练、正式训练、核验及报告入口。
# 关键副作用：构建本地 Docker 镜像，实验默认写入独立 Spark 目录。
set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Usage: bash SPARK.sh [--dry-run] <command> [arguments...]

Commands:
  setup   Build the ARM64 NVIDIA image, then run the GPU readiness check.
  check   Check the image, CUDA, frozen split, dependencies and GPU training.
  smoke   Run one epoch of mixed/dual with seed 42 in results/resnet18_spark_smoke.
  train   Run the official ResNet18 15-run fusion study (10 epochs, 6 threads), on CUDA.
  verify  Verify completed runs in results/study_resnet18_spark_v1.
  report  Summarize that study into report/c_part_resnet18_spark.
  tests   Run the project unit tests inside the same GPU container.

Extra arguments are forwarded to the Python entry point; later options override
the defaults (for example: train --output-dir results/my_run --resume).
Use paths relative to the project or /workspace inside the container.
SPARK_IMAGE overrides the default local image ds4023-c-resnet18-spark:25.11.
--dry-run works without Linux, Docker or a GPU. The image ID is a placeholder
until docker image inspect resolves it during an actual run.
USAGE
}

fail() { printf 'Spark launcher: %s\n' "$*" >&2; exit 1; }
print_command() { printf '%q ' "$@"; printf '\n'; }

dry_run=false
command_name=''
forward=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=true ;;
        -h|--help) usage; exit 0 ;;
        *)
            if [[ -z "$command_name" ]]; then
                command_name="$argument"
            else
                forward+=("$argument")
            fi
            ;;
    esac
done
if [[ -z "$command_name" ]]; then usage; exit 2; fi
case "$command_name" in
    setup|check|smoke|train|verify|report|tests) ;;
    *) printf 'Unknown command: %s\n' "$command_name" >&2; usage >&2; exit 2 ;;
esac

project_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_ref="${SPARK_IMAGE:-ds4023-c-resnet18-spark:25.11}"
image_id='sha256:RESOLVED_IMAGE_ID'
if ! "$dry_run"; then
    [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] ||
        fail 'Actual runs require Linux aarch64 on DGX Spark. Use --dry-run elsewhere.'
    command -v docker >/dev/null 2>&1 || fail 'Docker is unavailable; install the DGX Spark supported container runtime.'
    docker info >/dev/null || fail 'Docker daemon is unavailable or your user lacks permission. See DGX_SPARK.md.'
fi

if [[ "$command_name" == setup ]]; then
    build=(docker build --platform linux/arm64 --tag "$image_ref" "$project_root/spark")
    if "$dry_run"; then print_command "${build[@]}"; else "${build[@]}"; fi
    command_name=check
fi

inspection=(docker image inspect --format '{{.Architecture}}|{{.Id}}' "$image_ref")
if "$dry_run"; then
    print_command "${inspection[@]}"
else
    image_details="$("${inspection[@]}")" || fail "Image $image_ref is missing. Run bash SPARK.sh setup."
    image_arch="${image_details%%|*}"
    image_id="${image_details#*|}"
    [[ "$image_arch" == arm64 ]] || fail "Image architecture is $image_arch; linux/arm64 is required. Rebuild with setup."
    [[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'Docker returned an invalid image ID.'
fi

case "$command_name" in
    check)
        payload=(python tools/check_spark_environment.py --report
                 "results/spark_environments/${image_id#sha256:}.json") ;;
    smoke)
        payload=(python tools/run_spark_study.py --device cuda --epochs 1 --threads 6
                 --seeds 42 --inputs dual --augmentations mixed --output-dir results/resnet18_spark_smoke) ;;
    train|verify)
        training_entrypoint=tools/run_spark_study.py
        if [[ "$command_name" == verify ]]; then training_entrypoint=experiments/run_resnet_study.py; fi
        payload=(python "$training_entrypoint" --device cuda --epochs 10 --threads 6
                 --seeds 42 43 44 --output-dir results/study_resnet18_spark_v1)
        if [[ "$command_name" == verify ]]; then payload+=(--verify-only); fi ;;
    report)
        payload=(python experiments/summarize_resnet_study.py --study-dir results/study_resnet18_spark_v1
                 --report-dir report/c_part_resnet18_spark) ;;
    tests)
        payload=(python -m unittest discover -s experiments/tests -v) ;;
esac

run=(docker run --rm --init --platform linux/arm64 --gpus all --shm-size 1g
     --user "$(id -u):$(id -g)" --mount "type=bind,source=$project_root,target=/workspace"
     --workdir /workspace --env HOME=/tmp --env MPLCONFIGDIR=/tmp/mpl-cache --env MPLBACKEND=Agg
     --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1
     --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env NVIDIA_IMEX_CHANNELS=0
     --env NVIDIA_TF32_OVERRIDE=0 --env "SPARK_CONTAINER_IMAGE_ID=$image_id"
     --env "SPARK_CONTAINER_IMAGE_REF=$image_ref" "$image_id"
     "${payload[@]}" "${forward[@]}")
if "$dry_run"; then print_command "${run[@]}"; else exec "${run[@]}"; fi
