#!/usr/bin/env bash
# 外部评测记录驱动的反馈：单题诊断 → 供需交叉审计 → 下一轮 capabilities / curriculum。
# 配置放在项目根 .env 或环境变量中；命令行参数可覆盖默认值。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$PROJECT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT/.env"
  set +a
fi

# ─────────────── configurable parameters ───────────────
# External evaluator output: normalized/records.jsonl or a run directory.
RECORDS=(
  "$PROJECT/outputs/evaluation/records.jsonl"
)
BASE_CAPABILITIES="$PROJECT/outputs/capabilities.json"  # 最初的基础池
GEN_DIR="$PROJECT/outputs/task_gen"               # generate_tasks.sh 的 OUTPUT_DIR
EXTRA_GEN_DIRS=()                                              # 训练集里不在主目录的任务所在的其它 gen 目录
TRAIN_SET=()                                                   # 实际参训数据 jsonl（每行含 id）；留空则按整个题库统计
ROLLOUT_DIR="$PROJECT/outputs/rollout"  # 教师 rollout+judge；留空则不统计 trace check
LOG_PATH=""                                                    # 无 synthesis.jsonl；漏斗走 generation_progress.json（seeds 已带 archetype）
ROUND=1
PREV_ROUND_DIR=""   # 上一轮反馈目录；第一轮留空
OUT_DIR="$PROJECT/outputs/feedback/round_${ROUND}"
WORKERS=8
BLIND=0                                              # 1 = 忽略 eval_reason，强制盲诊
BASE_URL="${JUDGE_BASE_URL:-}"
API_KEY_FILE="${FEEDBACK_API_KEY_FILE:-}"
MODEL="${JUDGE_MODEL:-}"
# ────────────────────────────────────────

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec python3 -m evosynth.feedback.run --help
fi
if [[ ! -s "$API_KEY_FILE" ]]; then
  echo "密钥文件不存在或为空: $API_KEY_FILE" >&2
  exit 2
fi

cmd=(
  python3 -m evosynth.feedback.run
  --capabilities "$BASE_CAPABILITIES"
  --gen-dir "$GEN_DIR" "${EXTRA_GEN_DIRS[@]}"
  --out-dir "$OUT_DIR"
  --round "$ROUND"
  --workers "$WORKERS"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --api-key-file "$API_KEY_FILE"
)
if [[ -n "${RECORDS[0]}" ]]; then cmd+=(--records "${RECORDS[@]}"); fi
if [[ ${#TRAIN_SET[@]} -gt 0 ]]; then cmd+=(--train-set "${TRAIN_SET[@]}"); fi
if [[ -n "$ROLLOUT_DIR" ]]; then cmd+=(--rollout-dir "$ROLLOUT_DIR"); fi
if [[ -n "$LOG_PATH" ]]; then cmd+=(--log-path "$LOG_PATH"); fi
if [[ -n "$PREV_ROUND_DIR" ]]; then cmd+=(--prev-round-dir "$PREV_ROUND_DIR"); fi
if [[ "$BLIND" == "1" ]]; then cmd+=(--blind); fi

mkdir -p "$OUT_DIR"
exec "${cmd[@]}" "$@"

# To use this round for another synthesis pass, point generate_tasks.sh at
# $OUT_DIR/capabilities.json and set CURRICULUM_JSON to $OUT_DIR/curriculum.json.
