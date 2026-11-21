#!/usr/bin/env bash
# EvoSynth teacher rollout. Configuration may be provided in .env or the
# environment; command-line arguments override defaults.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"
# Ensure the local src tree is importable without an editable installation.
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$PROJECT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT/.env"
  set +a
fi

TASKS_DIR="${TASKS_DIR:-$PROJECT/outputs/task_gen/tasks}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/outputs/rollout}"
WORKERS="${WORKERS:-100}"
LIMIT="${LIMIT:-}"                                    # 空表示全量
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"                # 1 跳过已完成并断点续跑；0 全量重跑
REJUDGE_ERRORS="${REJUDGE_ERRORS:-0}"                # 1 只重跑 answer_judge_error，不重做 infer
BASE_URL="${LLMROUTER_BASE_URL:-}"
API_KEY_FILE="${EVOSYNTH_AGENT_API_KEY_FILE:-}"
MODEL="${INFER_MODEL:-}"
export EVOSYNTH_WORLD_MATERIALIZATION_RETRIES="${EVOSYNTH_WORLD_MATERIALIZATION_RETRIES:-3}"
# ────────────────────────────────────────

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec python3 -m evosynth.task_infer.rollout_tasks --help
fi

if [[ ! -d "$TASKS_DIR" ]]; then
  echo "任务目录不存在: $TASKS_DIR" >&2
  exit 2
fi
if [[ ! -s "$API_KEY_FILE" ]]; then
  echo "密钥文件不存在或为空: $API_KEY_FILE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
  python3 -m evosynth.task_infer.rollout_tasks
  --tasks-dir "$TASKS_DIR"
  --output-dir "$OUTPUT_DIR"
  --workers "$WORKERS"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --api-key-file "$API_KEY_FILE"
)
if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi
if [[ "$SKIP_COMPLETED" == "0" ]]; then
  cmd+=(--no-skip-completed)
fi
if [[ "$REJUDGE_ERRORS" == "1" ]]; then
  cmd+=(--stage judge --rejudge-errors)
fi

exec "${cmd[@]}" "$@"
