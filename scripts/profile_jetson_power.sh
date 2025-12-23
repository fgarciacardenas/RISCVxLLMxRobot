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
  --workdir PATH          Workdir inside container (default: /embodiedai/src/LLMxRobot)
  --cmd CMD               Command to run inside container (default: ./run_decision_tests_gpu_gguf.sh)
  --interval-ms MS        tegrastats interval in ms (default: 200)
  --baseline-s SEC        Baseline duration before workload (default: 10)
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

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" != "true" ]]; then
  echo "Container '$CONTAINER' is not running (or not found). Start it first, then retry." >&2
  exit 1
fi

container_has_dir() {
  local dir="$1"
  docker exec -i "$CONTAINER" bash -lc "test -d '$dir'" >/dev/null 2>&1
}

container_has_file() {
  local path="$1"
  docker exec -i "$CONTAINER" bash -lc "test -f '$path'" >/dev/null 2>&1
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

HAS_SETSID="0"
if command -v setsid >/dev/null 2>&1; then
  HAS_SETSID="1"
fi

terminate_pid() {
  local pid="${1:-}"
  local label="${2:-process}"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" >/dev/null 2>&1 || return 0

  if [[ "$HAS_SETSID" == "1" ]]; then
    # The pid should be a session leader; kill the whole group just in case.
    kill -TERM -- "-$pid" >/dev/null 2>&1 || true
  else
    kill -TERM "$pid" >/dev/null 2>&1 || true
  fi

  local deadline
  deadline="$(awk -v now="$(date +%s)" -v t="$KILL_TIMEOUT_S" 'BEGIN{printf "%d", now + (t+0)}')"
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
    else
      kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  fi
}

ENV_ARGS=()
for kv in "${ENV_KVS[@]}"; do
  ENV_ARGS+=(-e "$kv")
done
if [[ "$SEGMENT_LLM" == "1" ]]; then
  # Ensure event markers are enabled unless the user explicitly set it.
  has_marker_env="0"
  for kv in "${ENV_KVS[@]}"; do
    if [[ "$kv" == LLMXROBOT_PROFILE_LLM=* ]]; then
      has_marker_env="1"
      break
    fi
  done
  if [[ "$has_marker_env" == "0" ]]; then
    ENV_ARGS+=(-e "LLMXROBOT_PROFILE_LLM=1")
  fi
fi

CLEANED_UP="0"
INTERRUPTED="0"
cleanup() {
  [[ "$CLEANED_UP" == "1" ]] && return 0
  CLEANED_UP="1"

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

if [[ "$BASELINE_S" != "0" ]]; then
  echo "Baseline: ${BASELINE_S}s (keep system idle for best attribution)..."
  sleep "$BASELINE_S" || true
fi

if [[ "$INTERRUPTED" == "1" ]]; then
  echo "Skipping workload (interrupted during baseline)."
  WORKLOAD_EXIT=130
else
  echo "Running workload in container '$CONTAINER'..."

  # Run workload with a real PID we can stop, while still streaming logs.
  rm -f "$RUN_LOG"
  if [[ "$HAS_SETSID" == "1" ]]; then
    setsid docker exec "${ENV_ARGS[@]}" -i "$CONTAINER" bash -lc "cd '$WORKDIR' && $CMD" >"$RUN_LOG" 2>&1 &
  else
    docker exec "${ENV_ARGS[@]}" -i "$CONTAINER" bash -lc "cd '$WORKDIR' && $CMD" >"$RUN_LOG" 2>&1 &
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

echo "Stopping tegrastats..."
cleanup
trap - EXIT INT TERM

dt_s="$(awk -v ms="$INTERVAL_MS" 'BEGIN{printf "%.6f", ms/1000.0}')"
baseline_samples="$(awk -v b="$BASELINE_S" -v dt="$dt_s" 'BEGIN{if(dt<=0){print 0; exit} printf "%d", int((b/dt)+0.5)}')"

echo
echo "Summary (dt=${dt_s}s, baseline_samples=${baseline_samples}):"
echo "  - 'total' uses all samples (baseline + run)"
echo "  - 'run' excludes the baseline window"
echo "  - 'delta_energy_J' subtracts baseline mean power from run window"
echo

IFS=',' read -r -a rails <<<"$RAILS_CSV"
for rail in "${rails[@]}"; do
  rail="$(echo "$rail" | xargs)"
  [[ -n "$rail" ]] || continue
  awk -v rail="$rail" -v dt="$dt_s" -v skip="$baseline_samples" '
    BEGIN{
      n=0; sum=0; max=0;
      bn=0; bsum=0; bmax=0;
      rn=0; rsum=0; rmax=0;
    }
    {
      if (match($0, rail "[[:space:]]+([0-9]+)mW", m)) {
        p = m[1] + 0;
        n++; sum += p; if (p > max) max = p;
        if (n <= skip) { bn++; bsum += p; if (p > bmax) bmax = p; }
        else { rn++; rsum += p; if (p > rmax) rmax = p; }
      }
    }
    END{
      if (n == 0) { printf "%s: no samples found in tegrastats log\n", rail; exit 2; }
      dur = n * dt;
      printf "%s total: samples=%d dur_s=%.3f avg_mW=%.1f peak_mW=%d energy_J=%.3f\n",
             rail, n, dur, (sum/n), max, (sum*dt)/1000.0;
      if (skip > 0 && bn > 0) {
        bavg = bsum / bn;
        rdur = rn * dt;
        ravg = (rn > 0) ? (rsum / rn) : 0;
        deltaE = ((rsum - (bavg * rn)) * dt) / 1000.0;
        printf "%s run:   samples=%d dur_s=%.3f avg_mW=%.1f peak_mW=%d energy_J=%.3f baseline_mW=%.1f delta_energy_J=%.3f\n",
               rail, rn, rdur, ravg, rmax, (rsum*dt)/1000.0, bavg, deltaE;
      }
    }
  ' "$TEGRA_LOG"
done

if [[ "$SEGMENT_LLM" == "1" ]]; then
  echo
  echo "LLM-only segment summary (from workload event markers):"
  SEG_CSV="$LOG_DIR/llm_segments_${ts}.csv"
  python3 scripts/summarize_tegrastats_segments.py \
    --tegrastats-log "$TEGRA_LOG" \
    --run-log "$RUN_LOG" \
    --interval-ms "$INTERVAL_MS" \
    --baseline-samples "$baseline_samples" \
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
