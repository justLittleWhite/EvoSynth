#!/usr/bin/env bash
# EvoSynth task synthesis. Configuration may be provided in .env or the
# environment; command-line arguments are appended and override defaults.
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

# tool_info is an MCP pool grouped by domain. With SERVERS empty, the pipeline
# selects an anchor server and optional companions for each initial world. To
# use fixed servers, set SERVERS (for example, SERVERS=(google-maps)).

# capability_split 生成的能力文件路径（按需修改）
CAPABILITIES="$PROJECT/outputs/capabilities.json"
COUNT="${COUNT:-12000}"                                 # 成功发布的任务数量
SEED_START="${SEED_START:-20260806}"
WORKERS="${WORKERS:-90}"
# 每个 initial world 最多采样的 task 数，范围 1-10。实体更少时按实体数造，
# 因为每个 task 要独占一个起点实体。
WORLD_SAMPLES_PER_INITIAL_WORLD="${WORLD_SAMPLES_PER_INITIAL_WORLD:-3}"
OUTPUT_DIR="${EVOSYNTH_OUTPUT_ROOT:-$PROJECT/outputs/task_gen}"

TOOL_INFO="$PROJECT/configs/mcp_pool.json"
COMPANIONS=8
SERVERS=()
# 反馈闭环产物（可选）：run_feedback.sh 的输出目录里的 curriculum.json；留空表示不启用。
# 启用时把 CURRICULUM_JSON 指向对应的 curriculum.json。
CURRICULUM_JSON="${CURRICULUM_JSON:-}"
LOG_PATH="${LOG_PATH:-}"                                # 完整合成 JSONL 日志；反馈统计 task_audit_pass_rate 要用
BASE_URL="${LLMROUTER_BASE_URL:-}"
MODEL="${SYNTHESIS_MODEL:-}"
API_KEY_FILE="${EVOSYNTH_AGENT_API_KEY_FILE:-}"
WORLD_BASE_URL="${EVOSYNTH_WORLD_MODEL_URL:-}"
WORLD_MODEL="${EVOSYNTH_WORLD_MODEL:-}"
WORLD_API_KEY_FILE="${EVOSYNTH_WORLD_MODEL_API_KEY_FILE:-}"


# 之后相邻请求至少间隔 STEADY。遇到 Throttling.BurstRate 时优先调大 RAMP。
export EVOSYNTH_RAMP_INTERVAL_SECONDS="${EVOSYNTH_RAMP_INTERVAL_SECONDS:-1.5}"
export EVOSYNTH_STEADY_INTERVAL_SECONDS="${EVOSYNTH_STEADY_INTERVAL_SECONDS:-0.5}"
export EVOSYNTH_WORLD_RATE_LIMIT_RETRIES="${EVOSYNTH_WORLD_RATE_LIMIT_RETRIES:-3}"
export EVOSYNTH_WORLD_MATERIALIZATION_RETRIES="${EVOSYNTH_WORLD_MATERIALIZATION_RETRIES:-3}"
# Print concise stage/progress updates. Per-task construction traces are written
# next to each published task JSON.
export EVOSYNTH_STDOUT_PROGRESS="${EVOSYNTH_STDOUT_PROGRESS:-1}"
export EVOSYNTH_WORLD_SAMPLES_PER_INITIAL_WORLD="$WORLD_SAMPLES_PER_INITIAL_WORLD"
# 发布失败会用同一 seed 重跑 research 两次；仍失败则写入这份名单，之后不再使用。
# 与 fill_quota.sh 共用。
export EVOSYNTH_BANNED_SEEDS="${EVOSYNTH_BANNED_SEEDS:-$PROJECT/outputs/banned_seeds.json}"
export EVOSYNTH_PUBLISH_REBUILD_ATTEMPTS="${EVOSYNTH_PUBLISH_REBUILD_ATTEMPTS:-2}"
# ────────────────────────────────────────

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec python3 -m evosynth.task_gen.generate_tasks --help
fi

if [[ ! -f "$TOOL_INFO" ]]; then
  echo "tool_info 不存在: $TOOL_INFO" >&2
  exit 2
fi
if [[ ! -f "$CAPABILITIES" ]]; then
  echo "capabilities 不存在: $CAPABILITIES" >&2
  exit 2
fi
if [[ ! -s "$API_KEY_FILE" || ! -s "$WORLD_API_KEY_FILE" ]]; then
  echo "密钥文件不存在或为空: $API_KEY_FILE / $WORLD_API_KEY_FILE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
  python3 -m evosynth.task_gen.generate_tasks
  --tool-info "$TOOL_INFO"
  --capabilities "$CAPABILITIES"
  --companions "$COMPANIONS"
  --output-dir "$OUTPUT_DIR"
  --count "$COUNT"
  --seed-start "$SEED_START"
  --workers "$WORKERS"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --api-key-file "$API_KEY_FILE"
  --world-base-url "$WORLD_BASE_URL"
  --world-model "$WORLD_MODEL"
  --world-api-key-file "$WORLD_API_KEY_FILE"
)
if [[ ${#SERVERS[@]} -gt 0 ]]; then
  cmd+=(--servers "${SERVERS[@]}")
fi
if [[ -n "$LOG_PATH" ]]; then
  cmd+=(--log-path "$LOG_PATH")
fi
if [[ -n "$CURRICULUM_JSON" ]]; then
  export CURRICULUM_JSON
fi

exec "${cmd[@]}" "$@"
