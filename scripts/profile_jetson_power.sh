#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Profile Jetson power during a Docker workload using tegrastats.

Jetson does not expose true per-PID power; this reports rail power during the run
and (optionally) a baseline-subtracted "delta energy" as the closest practical
estimate of the workload's incremental consumption.

Usage:
  ./scripts/profile_jetson_power.sh [options]

Options:
  --container NAME        Docker container name (default: embodiedai_dock)
  --workdir PATH          Workdir inside container (default: /embodiedai)
  --cmd CMD               Command to run inside container (default: ./run_decision_tests_gpu_gguf.sh)
  --interval-ms MS        tegrastats interval in ms (default: 200)
  --baseline-s SEC        Baseline duration before workload (default: 10)
  --baseline-estimator X  Baseline estimator for delta power/energy (mean|p10|p50|min; default: p10)
  --rails A,B,C           Comma-separated rails to summarize (default: VIN_SYS_5V0,VDD_GPU_SOC,VDD_CPU_CV)
  --log-dir PATH          Host log output dir (default: src/LLMxRobot/logs/power_profiles)
  --env KEY=VAL           Pass env var into `docker exec` (repeatable)
  --segment-llm           Also compute energy for LLM decode windows only (requires LLM marker events)
  --segment-start EVENT   Start event name (default: llm_decode_start)
  --segment-end EVENT     End event name (default: llm_decode_end)
  --kill-timeout-s SEC    Seconds to wait before SIGKILL on stop (default: 5)
  -h, --help              Show this help

Example:
  ./scripts/profile_jetson_power.sh --baseline-s 15 --interval-ms 200
EOF
}

CONTAINER="embodiedai_dock"
WORKDIR="/embodiedai"
CMD="./run_decision_tests_gpu_gguf.sh"
INTERVAL_MS="200"
BASELINE_S="10"
BASELINE_ESTIMATOR="p10"
RAILS_CSV="VIN_SYS_5V0,VDD_GPU_SOC,VDD_CPU_CV"
LOG_DIR="src/LLMxRobot/logs/power_profiles"
SEGMENT_LLM="0"
SEGMENT_START_EVENT="llm_decode_start"
SEGMENT_END_EVENT="llm_decode_end"
ENV_KVS=()
KILL_TIMEOUT_S="5"

WORKDIR_SET_BY_USER="0"
CMD_SET_BY_USER="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) CONTAINER="${2:?}"; shift 2 ;;
    --workdir) WORKDIR="${2:?}"; WORKDIR_SET_BY_USER="1"; shift 2 ;;
    --cmd) CMD="${2:?}"; CMD_SET_BY_USER="1"; shift 2 ;;
    --interval-ms) INTERVAL_MS="${2:?}"; shift 2 ;;
    --baseline-s) BASELINE_S="${2:?}"; shift 2 ;;
    --baseline-estimator) BASELINE_ESTIMATOR="${2:?}"; shift 2 ;;
    --rails) RAILS_CSV="${2:?}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    --env) ENV_KVS+=("${2:?}"); shift 2 ;;
    --segment-llm) SEGMENT_LLM="1"; shift 1 ;;
    --segment-start) SEGMENT_START_EVENT="${2:?}"; shift 2 ;;
    --segment-end) SEGMENT_END_EVENT="${2:?}"; shift 2 ;;
    --kill-timeout-s) KILL_TIMEOUT_S="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

command -v tegrastats >/dev/null 2>&1 || { echo "tegrastats not found on host PATH." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker not found on host PATH." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found on host PATH." >&2; exit 1; }

# tegrastats typically requires sudo; refresh credentials up front so we don't block mid-run.
if ! sudo -n true >/dev/null 2>&1; then
  echo "Sudo is required to run tegrastats; prompting once now..." >&2
  sudo -v
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" != "true" ]]; then
  echo "Container '$CONTAINER' is not running (or not found). Start it first, then retry." >&2
  exit 1
fi

container_has_dir() {
  local dir="$1"
  docker exec "$CONTAINER" bash -lc "test -d '$dir'" >/dev/null 2>&1
}

container_has_file() {
  local path="$1"
  docker exec "$CONTAINER" bash -lc "test -f '$path'" >/dev/null 2>&1
}

container_log_dir_from_host() {
  # Best-effort mapping for the Jetson setup where host `src/LLMxRobot` is mounted at `/embodiedai`.
  # Returns empty string if we can't confidently map.
  if ! container_has_dir "/embodiedai"; then
    return 0
  fi
  case "$LOG_DIR" in
    src/LLMxRobot/*)
      echo "/embodiedai/${LOG_DIR#src/LLMxRobot/}"
      ;;
    logs/*)
      echo "/embodiedai/$LOG_DIR"
      ;;
    *)
      echo ""
      ;;
  esac
}

if [[ "$WORKDIR_SET_BY_USER" == "1" ]]; then
  if ! container_has_dir "$WORKDIR"; then
    echo "Workdir not found in container: $WORKDIR" >&2
    echo "Try --workdir /embodiedai (LLMxRobot is mounted there on Jetson by default)." >&2
    exit 1
  fi
else
  if container_has_dir "/embodiedai"; then
    WORKDIR="/embodiedai"
  elif container_has_dir "/embodiedai/src/LLMxRobot"; then
    WORKDIR="/embodiedai/src/LLMxRobot"
  fi
fi

if [[ "$CMD_SET_BY_USER" == "0" ]]; then
  if container_has_file "$WORKDIR/run_decision_tests_gpu_gguf.sh"; then
    CMD="./run_decision_tests_gpu_gguf.sh"
  elif container_has_file "$WORKDIR/src/LLMxRobot/run_decision_tests_gpu_gguf.sh"; then
    CMD="bash src/LLMxRobot/run_decision_tests_gpu_gguf.sh"
  fi
fi

mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
TEGRA_LOG="$LOG_DIR/tegrastats_${ts}.log"
RUN_LOG="$LOG_DIR/workload_${ts}.log"
: > "$RUN_LOG" || true

emit_runlog_event() {
  local ev="${1:?}"; shift || true
  local extra="${1:-}"
  local now
  now="$(python3 - <<'PY'\nimport time\nprint(f\"{time.time():.6f}\")\nPY\n)"
  if [[ -n "$extra" ]]; then
    printf 'LLMXROBOT_EVENT {"event":"%s","t_epoch_s":%s,%s}\n' "$ev" "$now" "$extra" >>"$RUN_LOG"
  else
    printf 'LLMXROBOT_EVENT {"event":"%s","t_epoch_s":%s}\n' "$ev" "$now" >>"$RUN_LOG"
  fi
}

HAS_SETSID="0"
if command -v setsid >/dev/null 2>&1; then
  HAS_SETSID="1"
fi

HAS_TIMEOUT="0"
if command -v timeout >/dev/null 2>&1; then
  HAS_TIMEOUT="1"
fi

terminate_pid() {
  local pid="${1:-}"
  local label="${2:-process}"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" >/dev/null 2>&1 || return 0

  # Try process group kill first (when started via setsid), then direct PID kill.
  if [[ "$HAS_SETSID" == "1" ]]; then
    kill -TERM -- "-$pid" >/dev/null 2>&1 || true
  fi
  kill -TERM "$pid" >/dev/null 2>&1 || true

  local deadline
  deadline=$(( $(date +%s) + KILL_TIMEOUT_S ))
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

ENV_ARGS=()
for kv in "${ENV_KVS[@]}"; do
  ENV_ARGS+=(-e "$kv")
done

# Avoid PermissionErrors / root-owned __pycache__ issues and ensure prompt-by-prompt progress
# is visible immediately in the host log.
has_no_pyc_env="0"
has_unbuffered_env="0"
for kv in "${ENV_KVS[@]}"; do
  if [[ "$kv" == PYTHONDONTWRITEBYTECODE=* ]]; then
    has_no_pyc_env="1"
  fi
  if [[ "$kv" == PYTHONUNBUFFERED=* ]]; then
    has_unbuffered_env="1"
  fi
done
if [[ "$has_no_pyc_env" == "0" ]]; then
  ENV_ARGS+=(-e "PYTHONDONTWRITEBYTECODE=1")
fi
if [[ "$has_unbuffered_env" == "0" ]]; then
  ENV_ARGS+=(-e "PYTHONUNBUFFERED=1")
fi

if [[ "$SEGMENT_LLM" == "1" ]]; then
  # Ensure event markers are enabled unless the user explicitly set it.
  has_marker_env="0"
  has_hard_exit_env="0"
  for kv in "${ENV_KVS[@]}"; do
    if [[ "$kv" == LLMXROBOT_PROFILE_LLM=* ]]; then
      has_marker_env="1"
    fi
    if [[ "$kv" == LLMXROBOT_HARD_EXIT=* ]]; then
      has_hard_exit_env="1"
    fi
  done
  if [[ "$has_marker_env" == "0" ]]; then
    ENV_ARGS+=(-e "LLMXROBOT_PROFILE_LLM=1")
  fi
  # On some Jetson/llama-cpp builds, Python can abort during interpreter shutdown.
  # Hard-exit avoids that so profiling still completes and CSVs are written.
  if [[ "$has_hard_exit_env" == "0" ]]; then
    ENV_ARGS+=(-e "LLMXROBOT_HARD_EXIT=1")
  fi
fi

CLEANED_UP="0"
INTERRUPTED="0"
PID_FILE_IN_CONTAINER=""
cleanup() {
  [[ "$CLEANED_UP" == "1" ]] && return 0
  CLEANED_UP="1"

  if [[ -n "$PID_FILE_IN_CONTAINER" ]]; then
    # Best-effort: stop the in-container workload process (killing host-side `docker exec`
    # alone may leave the workload running inside the container).
    kill_cmd="if [ -f '$PID_FILE_IN_CONTAINER' ]; then pid=\$(cat '$PID_FILE_IN_CONTAINER' 2>/dev/null || true); if [ -n \"\$pid\" ]; then kill -TERM -- -\"\$pid\" >/dev/null 2>&1 || kill -TERM \"\$pid\" >/dev/null 2>&1 || true; sleep 0.5; kill -0 \"\$pid\" >/dev/null 2>&1 && kill -KILL -- -\"\$pid\" >/dev/null 2>&1 || kill -KILL \"\$pid\" >/dev/null 2>&1 || true; fi; fi"
    if [[ "$HAS_TIMEOUT" == "1" ]]; then
      timeout 3s docker exec "$CONTAINER" bash -lc "$kill_cmd" </dev/null >/dev/null 2>&1 || true
    else
      docker exec "$CONTAINER" bash -lc "$kill_cmd" </dev/null >/dev/null 2>&1 || true
    fi
  fi

  terminate_pid "${TAIL_PID:-}" "tail"
  terminate_pid "${WORKLOAD_PID:-}" "docker exec workload"

  if [[ -n "${TEGRA_PID:-}" ]] && kill -0 "$TEGRA_PID" 2>/dev/null; then
    sudo kill -INT "$TEGRA_PID" >/dev/null 2>&1 || true
    # If sudo is the parent, also try to stop its child process.
    child_pid="$(pgrep -P "$TEGRA_PID" 2>/dev/null | head -n 1 || true)"
    if [[ -n "$child_pid" ]]; then
      sudo kill -INT "$child_pid" >/dev/null 2>&1 || true
    fi
    # Avoid hanging indefinitely if sudo/tegrastats ignores signals.
    terminate_pid "$TEGRA_PID" "tegrastats"
  fi
}

on_interrupt() {
  echo
  echo "Interrupted; stopping workload + tegrastats..."
  INTERRUPTED="1"
  cleanup
  # Do not exit here: allow the script to continue to the summary/CSV generation.
}

trap cleanup EXIT
trap on_interrupt INT TERM

echo "Logs:"
echo "  tegrastats: $TEGRA_LOG"
echo "  workload:   $RUN_LOG"
echo

echo "Starting tegrastats (${INTERVAL_MS}ms interval)..."
sudo tegrastats --interval "$INTERVAL_MS" --logfile "$TEGRA_LOG" &
TEGRA_PID=$!
emit_runlog_event "power_profile_start" "\"baseline_s\":${BASELINE_S},\"interval_ms\":${INTERVAL_MS},\"baseline_estimator\":\"${BASELINE_ESTIMATOR}\""

if [[ "$BASELINE_S" != "0" ]]; then
  echo "Baseline: ${BASELINE_S}s (keep system idle for best attribution)..."
  sleep "$BASELINE_S" || true
fi

if [[ "$INTERRUPTED" == "1" ]]; then
  echo "Skipping workload (interrupted during baseline)."
  WORKLOAD_EXIT=130
else
  echo "Running workload in container '$CONTAINER'..."
  emit_runlog_event "workload_start" "\"container\":\"${CONTAINER}\",\"workdir\":\"${WORKDIR}\""

  # Run workload detached inside the container, logging to the mounted filesystem.
  # This avoids `docker exec` sessions hanging after the in-container program finishes.
  DONE_FILE="$LOG_DIR/done_${ts}"
  EXIT_FILE="$LOG_DIR/exit_${ts}.txt"
  PID_FILE="$LOG_DIR/pid_${ts}.txt"
  rm -f "$DONE_FILE" "$EXIT_FILE" "$PID_FILE"

  LOG_DIR_IN_CONTAINER="$(container_log_dir_from_host)"
  if [[ -z "$LOG_DIR_IN_CONTAINER" ]]; then
    echo "Could not map host --log-dir to a container path; falling back to streaming docker exec output." >&2
    PID_FILE_IN_CONTAINER="/tmp/llmxrobot_profile_${ts}.pid"
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
  else
    RUN_LOG_IN_CONTAINER="$LOG_DIR_IN_CONTAINER/$(basename "$RUN_LOG")"
    DONE_FILE_IN_CONTAINER="$LOG_DIR_IN_CONTAINER/$(basename "$DONE_FILE")"
    EXIT_FILE_IN_CONTAINER="$LOG_DIR_IN_CONTAINER/$(basename "$EXIT_FILE")"
    PID_FILE_IN_CONTAINER="$LOG_DIR_IN_CONTAINER/$(basename "$PID_FILE")"

    # Start detached workload.
    RUN_ENV_ARGS=(
      "${ENV_ARGS[@]}"
      -e "LLMX_WORKDIR=$WORKDIR"
      -e "LLMX_CMD=$CMD"
      -e "LLMX_RUNLOG=$RUN_LOG_IN_CONTAINER"
      -e "LLMX_PIDFILE=$PID_FILE_IN_CONTAINER"
      -e "LLMX_EXITFILE=$EXIT_FILE_IN_CONTAINER"
      -e "LLMX_DONEFILE=$DONE_FILE_IN_CONTAINER"
    )
    docker exec "${RUN_ENV_ARGS[@]}" -d "$CONTAINER" bash -lc '
      cd "$LLMX_WORKDIR" || exit 1
      echo $$ > "$LLMX_PIDFILE"
      trap "kill 0" INT TERM HUP
      eval "$LLMX_CMD" >> "$LLMX_RUNLOG" 2>&1
      rc=$?
      printf "%s\n" "$rc" > "$LLMX_EXITFILE"
      touch "$LLMX_DONEFILE"
      exit "$rc"
    ' </dev/null >/dev/null 2>&1 || true

    # Stream logs from the host file as it grows.
    if [[ "$HAS_SETSID" == "1" ]]; then
      setsid tail -n +1 -f "$RUN_LOG" &
    else
      tail -n +1 -f "$RUN_LOG" &
    fi
    TAIL_PID=$!

    # Wait for completion marker (or interrupt).
    while [[ ! -f "$DONE_FILE" ]]; do
      if [[ "$INTERRUPTED" == "1" ]]; then
        break
      fi
      sleep 0.2
    done

    terminate_pid "$TAIL_PID" "tail"

    if [[ -f "$EXIT_FILE" ]]; then
      WORKLOAD_EXIT="$(cat "$EXIT_FILE" 2>/dev/null | tail -n 1 || true)"
    else
      WORKLOAD_EXIT=130
    fi
  fi
fi
emit_runlog_event "workload_end" "\"exit_code\":${WORKLOAD_EXIT:-130},\"interrupted\":${INTERRUPTED}"

echo "Stopping tegrastats..."
cleanup
trap - EXIT INT TERM

echo
echo "Summary (baseline_s=${BASELINE_S}s, interval_ms=${INTERVAL_MS}, baseline_estimator=${BASELINE_ESTIMATOR}):"
echo "  - 'total' uses all samples (baseline + run)"
echo "  - 'run' excludes the baseline window"
echo "  - 'delta_energy_J' subtracts baseline power estimate from run window"
echo

python3 scripts/summarize_tegrastats.py \
  --tegrastats-log "$TEGRA_LOG" \
  --interval-ms "$INTERVAL_MS" \
  --baseline-s "$BASELINE_S" \
  --baseline-estimator "$BASELINE_ESTIMATOR" \
  --rails "$RAILS_CSV" || true

if [[ "$SEGMENT_LLM" == "1" ]]; then
  echo
  echo "LLM-only segment summary (from workload event markers):"
  SEG_CSV="$LOG_DIR/llm_segments_${ts}.csv"
  python3 scripts/summarize_tegrastats_segments.py \
    --tegrastats-log "$TEGRA_LOG" \
    --run-log "$RUN_LOG" \
    --interval-ms "$INTERVAL_MS" \
    --baseline-s "$BASELINE_S" \
    --baseline-estimator "$BASELINE_ESTIMATOR" \
    --rails "$RAILS_CSV" \
    --start-event "$SEGMENT_START_EVENT" \
    --end-event "$SEGMENT_END_EVENT" \
    --out-csv "$SEG_CSV" || true
  echo "  segments_csv: $SEG_CSV"
fi

echo
if [[ "$INTERRUPTED" == "1" ]]; then
  echo "Exited early due to interrupt."
fi
exit "${WORKLOAD_EXIT:-130}"
