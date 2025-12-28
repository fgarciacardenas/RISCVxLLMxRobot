#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Profile NVIDIA GPU power (server) during a workload, and optionally compute LLM-only
energy for decode windows using LLMXROBOT_EVENT markers.

This is analogous to the Jetson tegrastats workflow, but uses GPU power sampling
via NVML (pynvml) if available, otherwise via nvidia-smi polling.

Usage:
  ./scripts/profile_nvidia_gpu_power.sh [options]

Options:
  --gpu IDX               GPU index to sample (default: 0)
  --interval-ms MS        Sampling interval in ms (default: 100)
  --baseline-s SEC        Baseline duration before workload (default: 10)
  --baseline-estimator X  Baseline estimator for delta power/energy (mean|p10|p50|min; default: p10)
  --log-dir PATH          Host log output dir (default: src/LLMxRobot/logs/power_profiles_nvidia)
  --cmd CMD               Workload command (default: python3 -m inference.gguf_decode_bench --help)
  --container NAME        If set, run workload inside this container via docker exec
  --workdir PATH          Workdir inside container (default: /embodiedai)
  --env KEY=VAL           Pass env var into docker exec (repeatable)
  --segment-llm           Also write llm_segments_*.csv from decode markers
  --segment-start EVENT   Start event (default: llm_decode_start)
  --segment-end EVENT     End event (default: llm_decode_end)
  --kill-timeout-s SEC    Seconds to wait before SIGKILL on stop (default: 5)
  -h, --help              Show help
EOF
}

GPU_IDX="0"
INTERVAL_MS="100"
BASELINE_S="10"
BASELINE_ESTIMATOR="p10"
LOG_DIR="src/LLMxRobot/logs/power_profiles_nvidia"
CMD="python3 -m inference.gguf_decode_bench --help"
CONTAINER=""
WORKDIR="/embodiedai"
ENV_KVS=()
SEGMENT_LLM="0"
SEGMENT_START_EVENT="llm_decode_start"
SEGMENT_END_EVENT="llm_decode_end"
KILL_TIMEOUT_S="5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_IDX="${2:?}"; shift 2 ;;
    --interval-ms) INTERVAL_MS="${2:?}"; shift 2 ;;
    --baseline-s) BASELINE_S="${2:?}"; shift 2 ;;
    --baseline-estimator) BASELINE_ESTIMATOR="${2:?}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    --cmd) CMD="${2:?}"; shift 2 ;;
    --container) CONTAINER="${2:?}"; shift 2 ;;
    --workdir) WORKDIR="${2:?}"; shift 2 ;;
    --env) ENV_KVS+=("${2:?}"); shift 2 ;;
    --segment-llm) SEGMENT_LLM="1"; shift 1 ;;
    --segment-start) SEGMENT_START_EVENT="${2:?}"; shift 2 ;;
    --segment-end) SEGMENT_END_EVENT="${2:?}"; shift 2 ;;
    --kill-timeout-s) KILL_TIMEOUT_S="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

command -v python3 >/dev/null 2>&1 || { echo "python3 not found." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found on PATH." >&2; exit 1; }

if [[ -n "$CONTAINER" ]]; then
  command -v docker >/dev/null 2>&1 || { echo "docker not found." >&2; exit 1; }
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" != "true" ]]; then
    echo "Container '$CONTAINER' is not running (or not found)." >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
POWER_LOG="$LOG_DIR/gpu_power_${ts}.csv"
RUN_LOG="$LOG_DIR/workload_${ts}.log"
SEG_CSV="$LOG_DIR/llm_segments_${ts}.csv"
PID_FILE_IN_CONTAINER=""
: > "$RUN_LOG" || true

HAS_SETSID="0"
command -v setsid >/dev/null 2>&1 && HAS_SETSID="1"
HAS_TIMEOUT="0"
command -v timeout >/dev/null 2>&1 && HAS_TIMEOUT="1"

terminate_pid() {
  local pid="${1:-}"
  local label="${2:-process}"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" >/dev/null 2>&1 || return 0

  if [[ "$HAS_SETSID" == "1" ]]; then
    kill -TERM -- "-$pid" >/dev/null 2>&1 || true
  fi
  kill -TERM "$pid" >/dev/null 2>&1 || true

  local deadline=$(( $(date +%s) + KILL_TIMEOUT_S ))
  while kill -0 "$pid" >/dev/null 2>&1; do
    if (( $(date +%s) >= deadline )); then
      break
    fi
    sleep 0.2
  done
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "Force killing $label (pid=$pid)..." >&2
    if [[ "$HAS_SETSID" == "1" ]]; then
      kill -KILL -- "-$pid" >/dev/null 2>&1 || true
    fi
    kill -KILL "$pid" >/dev/null 2>&1 || true
  fi
}

emit_runlog_event() {
  local ev="${1:?}"; shift || true
  local extra="${1:-}"
  local now
  now="$(python3 -c 'import time; print(f"{time.time():.6f}")')"
  if [[ -n "$extra" ]]; then
    printf 'LLMXROBOT_EVENT {"event":"%s","t_epoch_s":%s,%s}\n' "$ev" "$now" "$extra" >>"$RUN_LOG"
  else
    printf 'LLMXROBOT_EVENT {"event":"%s","t_epoch_s":%s}\n' "$ev" "$now" >>"$RUN_LOG"
  fi
}

ENV_ARGS=()
for kv in "${ENV_KVS[@]}"; do
  ENV_ARGS+=(-e "$kv")
done

if [[ "$SEGMENT_LLM" == "1" ]]; then
  # Ensure markers are enabled unless explicitly set.
  has_marker_env="0"
  for kv in "${ENV_KVS[@]}"; do
    [[ "$kv" == LLMXROBOT_PROFILE_LLM=* ]] && has_marker_env="1"
  done
  [[ "$has_marker_env" == "0" ]] && ENV_ARGS+=(-e "LLMXROBOT_PROFILE_LLM=1")
fi

cleanup() {
  if [[ -n "$CONTAINER" && -n "$PID_FILE_IN_CONTAINER" ]]; then
    kill_cmd="if [ -f '$PID_FILE_IN_CONTAINER' ]; then pid=\$(cat '$PID_FILE_IN_CONTAINER' 2>/dev/null || true); if [ -n \"\$pid\" ]; then kill -TERM -- -\"\$pid\" >/dev/null 2>&1 || kill -TERM \"\$pid\" >/dev/null 2>&1 || true; sleep 0.5; kill -0 \"\$pid\" >/dev/null 2>&1 && (kill -KILL -- -\"\$pid\" >/dev/null 2>&1 || kill -KILL \"\$pid\" >/dev/null 2>&1 || true); fi; fi"
    if [[ "$HAS_TIMEOUT" == "1" ]]; then
      timeout 3s docker exec "$CONTAINER" bash -lc "$kill_cmd" </dev/null >/dev/null 2>&1 || true
    else
      docker exec "$CONTAINER" bash -lc "$kill_cmd" </dev/null >/dev/null 2>&1 || true
    fi
  fi
  terminate_pid "${TAIL_PID:-}" "tail"
  terminate_pid "${WORKLOAD_PID:-}" "workload"
  terminate_pid "${LOGGER_PID:-}" "GPU power logger"
}

on_interrupt() {
  echo
  echo "Interrupted; stopping workload + logger..."
  INTERRUPTED="1"
  cleanup
}

trap cleanup EXIT
trap on_interrupt INT TERM
INTERRUPTED="0"

echo "Logs:"
echo "  gpu_power: $POWER_LOG"
echo "  workload:  $RUN_LOG"
echo

echo "Starting GPU power logger (gpu=$GPU_IDX, ${INTERVAL_MS}ms)..."
python3 scripts/log_nvidia_gpu_power.py --gpu "$GPU_IDX" --interval-ms "$INTERVAL_MS" --out-csv "$POWER_LOG" &
LOGGER_PID=$!
emit_runlog_event "power_profile_start" "\"gpu\":${GPU_IDX},\"baseline_s\":${BASELINE_S},\"interval_ms\":${INTERVAL_MS},\"baseline_estimator\":\"${BASELINE_ESTIMATOR}\""

if [[ "$BASELINE_S" != "0" ]]; then
  echo "Baseline: ${BASELINE_S}s..."
  sleep "$BASELINE_S" || true
fi

if [[ "$INTERRUPTED" == "1" ]]; then
  echo "Skipping workload (interrupted during baseline)."
  WORKLOAD_EXIT=130
else
  emit_runlog_event "workload_start" "\"container\":\"${CONTAINER}\",\"workdir\":\"${WORKDIR}\""
  echo "Running workload..."

  if [[ -z "$CONTAINER" ]]; then
    RUN_CMD="$CMD"
    if [[ "$SEGMENT_LLM" == "1" ]]; then
      RUN_CMD="LLMXROBOT_PROFILE_LLM=1 $RUN_CMD"
    fi
    if [[ "$HAS_SETSID" == "1" ]]; then
      setsid bash -lc "eval \"$RUN_CMD\"" </dev/null >>"$RUN_LOG" 2>&1 &
    else
      bash -lc "eval \"$RUN_CMD\"" </dev/null >>"$RUN_LOG" 2>&1 &
    fi
    WORKLOAD_PID=$!
    if [[ "$HAS_SETSID" == "1" ]]; then
      setsid tail -n +1 -f "$RUN_LOG" &
    else
      tail -n +1 -f "$RUN_LOG" &
    fi
    TAIL_PID=$!
    set +e
    wait "$WORKLOAD_PID"
    WORKLOAD_EXIT=$?
    set -e
    terminate_pid "$TAIL_PID" "tail"
  else
    # Streamed docker exec (avoids reliance on host-path mounts inside the container).
    PID_FILE_IN_CONTAINER="/tmp/llmxrobot_profile_gpu_${ts}.pid"
    RUN_ENV_ARGS=(
      "${ENV_ARGS[@]}"
      -e "LLMX_WORKDIR=$WORKDIR"
      -e "LLMX_CMD=$CMD"
      -e "LLMX_PIDFILE=$PID_FILE_IN_CONTAINER"
    )

    if [[ "$HAS_SETSID" == "1" ]]; then
      setsid docker exec "${RUN_ENV_ARGS[@]}" "$CONTAINER" bash -lc 'cd "$LLMX_WORKDIR" && echo $$ > "$LLMX_PIDFILE" && trap "kill 0" INT TERM HUP && eval "$LLMX_CMD"' </dev/null >>"$RUN_LOG" 2>&1 &
    else
      docker exec "${RUN_ENV_ARGS[@]}" "$CONTAINER" bash -lc 'cd "$LLMX_WORKDIR" && echo $$ > "$LLMX_PIDFILE" && trap "kill 0" INT TERM HUP && eval "$LLMX_CMD"' </dev/null >>"$RUN_LOG" 2>&1 &
    fi
    WORKLOAD_PID=$!

    if [[ "$HAS_SETSID" == "1" ]]; then
      setsid tail -n +1 -f "$RUN_LOG" &
    else
      tail -n +1 -f "$RUN_LOG" &
    fi
    TAIL_PID=$!

    set +e
    wait "$WORKLOAD_PID"
    WORKLOAD_EXIT=$?
    set -e

    terminate_pid "$TAIL_PID" "tail"
  fi
fi

emit_runlog_event "workload_end" "\"exit_code\":${WORKLOAD_EXIT:-130},\"interrupted\":${INTERRUPTED}"

echo "Stopping GPU power logger..."
cleanup
trap - EXIT INT TERM

echo
echo "Summary:"
python3 scripts/summarize_nvidia_gpu_power.py \
  --power-csv "$POWER_LOG" \
  --run-log "$RUN_LOG" \
  --baseline-s "$BASELINE_S" \
  --baseline-estimator "$BASELINE_ESTIMATOR" || true

echo
echo "LLM decode segment CSV:"
if [[ "$SEGMENT_LLM" == "1" ]]; then
  python3 scripts/summarize_nvidia_gpu_power_segments.py \
    --power-csv "$POWER_LOG" \
    --run-log "$RUN_LOG" \
    --out-csv "$SEG_CSV" \
    --gpu "$GPU_IDX" \
    --baseline-s "$BASELINE_S" \
    --baseline-estimator "$BASELINE_ESTIMATOR" \
    --start-event "$SEGMENT_START_EVENT" \
    --end-event "$SEGMENT_END_EVENT" || true
  echo "  segments_csv: $SEG_CSV"
else
  echo "  (skipped; pass --segment-llm)"
fi

echo
if [[ "$INTERRUPTED" == "1" ]]; then
  echo "Exited early due to interrupt."
fi
exit "${WORKLOAD_EXIT:-130}"
