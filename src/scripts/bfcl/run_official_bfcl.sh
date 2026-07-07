#!/usr/bin/env bash
# Generate and evaluate one BFCL test category for one model using the official
# BFCL harness (vLLM server backend).
#
# Prerequisites:
#   - A checkout of the BFCL repo (berkeley-function-call-leaderboard) with the
#     thesis model handlers registered via register_thesis_models.py.
#   - vLLM and the `bfcl` CLI installed in the current environment.
#
# Usage:
#   MODEL=Qwen/Qwen3-4B TEST_CATEGORY=simple_python \
#   BFCL_ROOT=/path/to/gorilla/berkeley-function-call-leaderboard \
#     bash src/scripts/bfcl/run_official_bfcl.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
TEST_CATEGORY="${TEST_CATEGORY:-simple_python}"
BFCL_ROOT="${BFCL_ROOT:?Set BFCL_ROOT to the berkeley-function-call-leaderboard checkout}"
RUN_ROOT="${RUN_ROOT:-$(pwd)/outputs/bfcl/official_runs/${TEST_CATEGORY}}"
PORT="${PORT:-1053}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TEMPERATURE="${TEMPERATURE:-0.7}"
RESULT_DIR="${RESULT_DIR:-$RUN_ROOT/result}"
SCORE_DIR="${SCORE_DIR:-$RUN_ROOT/score}"

SAFE_MODEL="$(printf '%s' "$MODEL" | tr '/:' '__')"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

cleanup() {
  pkill -f "vllm serve ${MODEL}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "$BFCL_ROOT"

echo "[bfcl] Starting vLLM server for ${MODEL} on port ${PORT}"
vllm serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  > "$LOG_DIR/${SAFE_MODEL}.vllm.log" 2>&1 &

for _ in $(seq 1 240); do
  if curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[bfcl] Server is ready"
    break
  fi
  sleep 2
done

curl -fsS "http://localhost:${PORT}/v1/models" > "$LOG_DIR/${SAFE_MODEL}.models.json"

echo "[bfcl] Generating ${TEST_CATEGORY} for ${MODEL}"
LOCAL_SERVER_ENDPOINT=localhost \
LOCAL_SERVER_PORT="$PORT" \
BFCL_PROJECT_ROOT="$BFCL_ROOT" \
bfcl generate \
  --model "$MODEL" \
  --test-category "$TEST_CATEGORY" \
  --backend vllm \
  --skip-server-setup \
  --temperature "$TEMPERATURE" \
  --result-dir "$RESULT_DIR" \
  --allow-overwrite \
  > "$LOG_DIR/${SAFE_MODEL}.generate.log" 2>&1

echo "[bfcl] Evaluating ${TEST_CATEGORY} for ${MODEL}"
BFCL_PROJECT_ROOT="$BFCL_ROOT" \
bfcl evaluate \
  --model "$MODEL" \
  --test-category "$TEST_CATEGORY" \
  --result-dir "$RESULT_DIR" \
  --score-dir "$SCORE_DIR" \
  > "$LOG_DIR/${SAFE_MODEL}.evaluate.log" 2>&1

echo "[bfcl] Done. Results: $RUN_ROOT"
