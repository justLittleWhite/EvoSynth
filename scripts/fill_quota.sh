#!/usr/bin/env bash
# 配额补缺循环：造题 → 教师 rollout+check → 数各桶 PASS 数 → 只补有缺口的桶，直到凑满。
# 最终得到 TARGET_TOTAL 道「已发布且 rollout PASS」的题，各合成对象数量严格等于配额。
# 先改下面「可改参数」，再直接运行；中断后重跑会从已有各轮继续。
# 命令行里再传的参数会追加在默认参数后面，可以覆盖同名项（例如 --dry-run）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

# ─────────────── 可改参数 ───────────────
FEEDBACK_DIR="${FEEDBACK_DIR:-$PROJECT/outputs/feedback/mcp_atlas_round_1}"       # feedback 输出目录（含 capabilities/curriculum/audit_report）
TARGET_TOTAL="${TARGET_TOTAL:-1250}"                                     # 最终要凑够的新题数（发布且 rollout PASS）
WORK_DIR="${WORK_DIR:-$PROJECT/outputs/mcp_atlas_fill_round_1}"    # 各轮产物与 final/ 都在这里
TOOL_INFO="${TOOL_INFO:-$PROJECT/configs/mcp_pool.json}"
WORKERS="${WORKERS:-30}"                                             # 造题并发
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-30}"
COMPANIONS="${COMPANIONS:-8}"
WORLD_SAMPLES_PER_INITIAL_WORLD="${WORLD_SAMPLES_PER_INITIAL_WORLD:-3}"
SEED_START="${SEED_START:-20281020}"
MARGIN="${MARGIN:-1.15}"                                            # 每轮申请量 = 缺口 × MARGIN ÷ 预估通过率
MAX_ROUNDS="${MAX_ROUNDS:-8}"
# 与 generate_tasks.sh 共用：发布失败重跑 research 两次，仍失败则封禁 seed。
export EVOSYNTH_BANNED_SEEDS="${EVOSYNTH_BANNED_SEEDS:-$PROJECT/outputs/banned_seeds.json}"
export EVOSYNTH_PUBLISH_REBUILD_ATTEMPTS="${EVOSYNTH_PUBLISH_REBUILD_ATTEMPTS:-2}"

BASE_URL=""
API_KEY_FILE=""
# URL must be provided by the user.
MODEL="${MODEL:-deepseek-v4.1-flash-1017}"
# API_KEY_FILE=""
WORLD_BASE_URL=""
WORLD_API_KEY_FILE=""
# URL must be provided by the user.
WORLD_MODEL="${WORLD_MODEL:-deepseek-v4.1-flash-1017}"
# WORLD_API_KEY_FILE=""
ROLLOUT_MODEL="${ROLLOUT_MODEL:-deepseek-v4.1-flash-1017}"                                # 教师 rollout 模型
# ────────────────────────────────────────

exec python3 -m evosynth.feedback.fill_quota \
  --feedback-dir "$FEEDBACK_DIR" --target-total "$TARGET_TOTAL" --work-dir "$WORK_DIR" \
  --tool-info "$TOOL_INFO" --workers "$WORKERS" --rollout-workers "$ROLLOUT_WORKERS" \
  --companions "$COMPANIONS" --world-samples-per-initial-world "$WORLD_SAMPLES_PER_INITIAL_WORLD" \
  --seed-start "$SEED_START" --margin "$MARGIN" --max-rounds "$MAX_ROUNDS" \
  --base-url "$BASE_URL" --model "$MODEL" --api-key-file "$API_KEY_FILE" \
  --world-base-url "$WORLD_BASE_URL" --world-model "$WORLD_MODEL" \
  --world-api-key-file "$WORLD_API_KEY_FILE" --rollout-model "$ROLLOUT_MODEL" \
  "$@"
