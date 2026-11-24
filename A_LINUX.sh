#!/usr/bin/env bash
# 用途：A 的 Linux GPU 服务器一站式入口：环境、校验、九次正式训练、十三条件评估与汇总。
# 关键约束：正式协议固定 raw 输入 × 三种增强策略 × 种子 42/43/44；脚本不改变 batch、轮数或精度。
# 用法：bash A_LINUX.sh {env|setup|check|smoke|train|resume|verify|eval13|report|freeze|test7|status|all}
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${A_PYTHON:-python}"
DEVICE="${A_DEVICE:-cuda}"
OUTPUT="${A_OUTPUT:-results/a_resnet18_linux_v1}"
SMOKE_OUTPUT="${A_SMOKE_OUTPUT:-results/a_linux_smoke_v1}"
EVAL_OUTPUT="${A_EVAL_OUTPUT:-results/a_eval13_linux_v1}"
REPORT_DIR="${A_REPORT_DIR:-report/a_part}"
THREADS="${A_THREADS:-6}"
DATA_ROOT="${A_DATA_ROOT:-data/GTSRB/raw}"
SPLIT="${A_SPLIT:-results/pilot_split.csv}"
WEIGHTS="${A_WEIGHTS:-assets/weights/resnet18-f37072fd.pth}"
WEIGHTS_SHA256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
FREEZE="${A_TEST_FREEZE:-results/a_test7_freeze.json}"
TEST_OUTPUT="${A_TEST_OUTPUT:-results/a_test7_v1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

TRAIN_DATA=(--data-root "$DATA_ROOT" --split "$SPLIT" --weights-path "$WEIGHTS")
COMMON=(--inputs raw --augmentations repeat_clean targeted_motion mixed
        --seeds 42 43 44 --epochs 10 --threads "$THREADS" --device "$DEVICE"
        --output-dir "$OUTPUT" "${TRAIN_DATA[@]}")

note() { printf '\n==> %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

require_python() {
  command -v "$PYTHON" >/dev/null 2>&1 || die "找不到 Python：$PYTHON。先运行 bash A_LINUX.sh setup，或设置 A_PYTHON。"
  "$PYTHON" - <<'PY' >/dev/null 2>&1 || die "当前 Python 缺少 torch。先运行 bash A_LINUX.sh setup。"
import torch
PY
}

require_data() {
  [[ -d "$DATA_ROOT/GTSRB/Final_Training/Images" ]] || die "缺少训练图像：$DATA_ROOT/GTSRB/Final_Training/Images"
  [[ -d "$DATA_ROOT/GTSRB/Final_Test/Images" ]] || die "缺少测试图像：$DATA_ROOT/GTSRB/Final_Test/Images"
  [[ -f "$DATA_ROOT/GT-final_test.csv" ]] || die "缺少官方测试标注：$DATA_ROOT/GT-final_test.csv"
  [[ -f "$SPLIT" ]] || die "缺少冻结划分：$SPLIT"
  [[ -f "$WEIGHTS" ]] || die "缺少官方 ResNet18 初始化：$WEIGHTS（见 assets/weights/README.md）"
}

require_weights() {
  if command -v sha256sum >/dev/null 2>&1; then
    local actual
    actual=$(sha256sum "$WEIGHTS" | awk '{print $1}')
    [[ "$actual" == "$WEIGHTS_SHA256" ]] || die "官方权重校验失败：$WEIGHTS 的 sha256 是 $actual"
  else
    printf 'WARNING: 没有 sha256sum，跳过权重校验。\n'
  fi
}

detect_torch_index() {
  local line major minor tag tmaj tmin
  line=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | head -1)
  [[ -n "$line" ]] || return 1
  major=${line% *}; minor=${line#* }
  for tag in 130 128 126 124 121 118; do
    tmaj=${tag:0:2}; tmin=${tag:2:1}
    if (( tmaj < major )) || { (( tmaj == major )) && (( tmin <= minor )); }; then
      printf '%s' "$tag"; return 0
    fi
  done
  return 1
}

case "${1:-help}" in
env)
  note "系统与硬件"
  uname -a || true
  nvidia-smi || printf 'WARNING: nvidia-smi 不可用。\n'
  command -v "$PYTHON" >/dev/null 2>&1 && "$PYTHON" -V || printf 'WARNING: 找不到 %s\n' "$PYTHON"
  free -h || true
  df -h . || true
  note "协议文件"
  for path in "$SPLIT" "$WEIGHTS" "$DATA_ROOT/GT-final_test.csv"; do
    if [[ -e "$path" ]]; then printf 'OK      %s\n' "$path"; else printf 'MISSING %s\n' "$path"; fi
  done
  ;;

setup)
  note "创建独立虚拟环境 .venv-a-linux"
  python3 -m venv .venv-a-linux
  # shellcheck disable=SC1091
  . .venv-a-linux/bin/activate
  python -m pip install --upgrade pip
  if [[ -n "${A_TORCH_INDEX:-}" ]]; then
    INDEX="$A_TORCH_INDEX"
  elif INDEX_TAG=$(detect_torch_index); then
    INDEX="https://download.pytorch.org/whl/cu${INDEX_TAG}"
  else
    die "无法从 nvidia-smi 推断 CUDA 版本。请到 https://pytorch.org/get-started/locally/ 选好后运行：
  A_TORCH_INDEX=https://download.pytorch.org/whl/cuXXX bash A_LINUX.sh setup"
  fi
  note "安装 Torch/Torchvision：$INDEX"
  python -m pip install torch torchvision --index-url "$INDEX"
  note "安装其余依赖（requirements-a.txt）"
  if ! python -m pip install -r requirements-a.txt; then
    printf 'WARNING: 固定版本在当前 Python 上没有可用 wheel，退回兼容范围。\n'
    printf 'WARNING: 跨机器的像素一致性必须用 data_fingerprint 逐项核对，不能假设相同。\n'
    python -m pip install 'numpy>=2.1,<3' 'opencv-python-headless>=4.10,<5' \
      'scikit-learn>=1.5,<1.10' 'matplotlib>=3.8,<3.12'
  fi
  python -m pip check
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() 仍为 False：换用匹配驱动的 CUDA wheel，不要继续。")
print("torch", torch.__version__, "CUDA runtime", torch.version.cuda,
      "GPU", torch.cuda.get_device_name(0))
PY
  note "环境已就绪。之后每个新终端先执行：source .venv-a-linux/bin/activate"
  ;;

check)
  require_python
  require_data
  require_weights
  note "GPU 与显存探针"
  "$PYTHON" tools/check_a_linux.py --weights "$WEIGHTS"
  note "数据协议与预算审计"
  "$PYTHON" tools/audit_a_data.py --data-root "$DATA_ROOT" --split "$SPLIT" \
    --output results/a_linux_environment/data_audit
  note "回归测试（数据协议、13 条件、汇总程序）"
  export A_DATA_ROOT="$DATA_ROOT" A_SPLIT="$SPLIT"
  "$PYTHON" -m unittest discover -s experiments/tests -p 'test_c_data_protocol.py' -v
  "$PYTHON" -m unittest discover -s experiments/tests -p 'test_a_*.py' -v
  note "检查通过。下一步：bash A_LINUX.sh smoke"
  ;;

smoke)
  require_python
  require_data
  require_weights
  note "工程试跑：mixed × 种子 42 × 1 轮（不计入正式结果）"
  "$PYTHON" experiments/run_resnet_study.py --inputs raw --augmentations mixed \
    --seeds 42 --epochs 1 --threads "$THREADS" --device "$DEVICE" --output-dir "$SMOKE_OUTPUT" \
    "${TRAIN_DATA[@]}"
  "$PYTHON" experiments/run_resnet_study.py --inputs raw --augmentations mixed \
    --seeds 42 --epochs 1 --threads "$THREADS" --device "$DEVICE" --output-dir "$SMOKE_OUTPUT" \
    "${TRAIN_DATA[@]}" --verify-only
  note "试跑通过。正式训练输出目录：$OUTPUT"
  ;;

train)
  require_python
  require_data
  require_weights
  if [[ -e "$OUTPUT" ]]; then
    die "$OUTPUT 已存在。换新目录（A_OUTPUT=...）或改用 bash A_LINUX.sh resume。"
  fi
  if [[ -z "${TMUX:-}" ]]; then
    printf 'WARNING: 建议在 tmux 里运行，断开 SSH 会中断训练。\n'
  fi
  note "正式九次训练：raw × 三种增强 × 种子 42/43/44，每次 10 轮"
  mkdir -p logs
  "$PYTHON" experiments/run_resnet_study.py "${COMMON[@]}" 2>&1 | tee logs/a_train.log
  note "训练结束。下一步：bash A_LINUX.sh verify"
  ;;

resume)
  require_python
  require_data
  note "续跑：只跳过已完整通过核验的运行"
  mkdir -p logs
  "$PYTHON" experiments/run_resnet_study.py "${COMMON[@]}" --resume 2>&1 | tee -a logs/a_train.log
  ;;

verify)
  require_python
  note "只读核验九个运行、配置、产物与指纹"
  "$PYTHON" experiments/run_resnet_study.py "${COMMON[@]}" --verify-only
  ;;

eval13)
  require_python
  [[ -d "$OUTPUT/runs" ]] || die "没有训练结果：$OUTPUT。先跑 bash A_LINUX.sh train。"
  note "开发集十三条件评估：加载最终权重，不重新训练"
  "$PYTHON" experiments/evaluate_a13.py --study "$OUTPUT" --output "$EVAL_OUTPUT" \
    --device "$DEVICE" --data-root "$DATA_ROOT" --split "$SPLIT" --expect-runs 9
  note "评估结束。下一步：bash A_LINUX.sh report"
  ;;

report)
  require_python
  [[ -f "$EVAL_OUTPUT/completion.json" ]] || die "没有评估结果：$EVAL_OUTPUT。先跑 bash A_LINUX.sh eval13。"
  note "生成十三条件表格、配对差值与强度曲线"
  "$PYTHON" experiments/summarize_a_study.py --eval "$EVAL_OUTPUT" --output "$REPORT_DIR"
  note "报告材料写入 $REPORT_DIR"
  ;;

freeze)
  require_python
  note "冻结官方测试的比较清单（在打开测试分数之前完成）"
  "$PYTHON" experiments/evaluate_official_test.py --init-freeze "$FREEZE" --study "$OUTPUT" \
    --data-root "$DATA_ROOT" --gt "$DATA_ROOT/GT-final_test.csv"
  note "冻结文件：$FREEZE。请勿在查看测试结果后修改它。"
  ;;

test7)
  require_python
  require_data
  [[ -f "$FREEZE" ]] || die "缺少冻结清单：$FREEZE。先跑 bash A_LINUX.sh freeze。"
  note "官方七类测试子集独立评估"
  "$PYTHON" experiments/evaluate_official_test.py --freeze "$FREEZE" --output "$TEST_OUTPUT" \
    --device "$DEVICE" --data-root "$DATA_ROOT" --gt "$DATA_ROOT/GT-final_test.csv"
  ;;

status)
  for dir in "$OUTPUT" "$SMOKE_OUTPUT" "$EVAL_OUTPUT" "$TEST_OUTPUT"; do
    if [[ ! -d "$dir" ]]; then
      printf '%-45s (missing)\n' "$dir"
      continue
    fi
    if [[ "$dir" == "$OUTPUT" || "$dir" == "$SMOKE_OUTPUT" ]]; then
      runs=0
      for marker in "$dir"/runs/*/completion.json; do
        if [[ -f "$marker" ]]; then
          runs=$((runs + 1))
        fi
      done
      state="$runs completed runs"
    elif [[ -f "$dir/completion.json" ]]; then
      state=complete
    else
      state=incomplete
    fi
    printf '%-45s %s, %s\n' "$dir" "$state" "$(du -sh "$dir" 2>/dev/null | cut -f1)"
  done
  ;;

all)
  for step in check smoke train verify eval13 report; do
    bash ./A_LINUX.sh "$step"
  done
  ;;

help|*)
  cat <<'USAGE'
A 的 Linux 入口：
  env      查看系统、GPU、磁盘与协议文件是否就位
  setup    创建 .venv-a-linux，安装匹配 CUDA 的 Torch/Torchvision 与其余依赖
  check    GPU/显存探针 + 数据预算审计 + 三组回归测试
  smoke    mixed × 种子 42 × 1 轮工程试跑（不计入正式结果）
  train    正式九次训练：raw × repeat_clean/targeted_motion/mixed × 42/43/44
  resume   续跑，只跳过已完整通过核验的运行
  verify   只读核验九个运行与全部产物指纹
  eval13   开发集十三条件评估（不重新训练）
  report   汇总表格、配对差值、强度曲线到 report/a_part
  freeze   写出官方测试评估的冻结清单
  test7    官方七类测试子集独立评估（需先 freeze）
  status   查看各输出目录的进度
  all      check → smoke → train → verify → eval13 → report

可用环境变量：A_PYTHON A_DEVICE A_THREADS A_OUTPUT A_SMOKE_OUTPUT A_EVAL_OUTPUT
              A_REPORT_DIR A_DATA_ROOT A_SPLIT A_WEIGHTS A_TORCH_INDEX A_TEST_FREEZE A_TEST_OUTPUT
USAGE
  ;;
esac
