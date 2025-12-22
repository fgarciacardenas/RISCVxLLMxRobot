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
  -h, --help              Show this help

Example:
  ./scripts/profile_jetson_power.sh --baseline-s 15 --interval-ms 200
EOF
}

CONTAINER="embodiedai_dock"
WORKDIR="/embodiedai/src/LLMxRobot"
CMD="./run_decision_tests_gpu_gguf.sh"
INTERVAL_MS="200"
BASELINE_S="10"
RAILS_CSV="VIN_SYS_5V0,VDD_GPU_SOC,VDD_CPU_CV"
LOG_DIR="src/LLMxRobot/logs/power_profiles"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) CONTAINER="${2:?}"; shift 2 ;;
    --workdir) WORKDIR="${2:?}"; shift 2 ;;
    --cmd) CMD="${2:?}"; shift 2 ;;
    --interval-ms) INTERVAL_MS="${2:?}"; shift 2 ;;
    --baseline-s) BASELINE_S="${2:?}"; shift 2 ;;
    --rails) RAILS_CSV="${2:?}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

command -v tegrastats >/dev/null 2>&1 || { echo "tegrastats not found on host PATH." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker not found on host PATH." >&2; exit 1; }

mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
TEGRA_LOG="$LOG_DIR/tegrastats_${ts}.log"
RUN_LOG="$LOG_DIR/workload_${ts}.log"

cleanup() {
  if [[ -n "${TEGRA_PID:-}" ]] && kill -0 "$TEGRA_PID" 2>/dev/null; then
    sudo kill -INT "$TEGRA_PID" >/dev/null 2>&1 || true
    wait "$TEGRA_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Logs:"
echo "  tegrastats: $TEGRA_LOG"
echo "  workload:   $RUN_LOG"
echo

echo "Starting tegrastats (${INTERVAL_MS}ms interval)..."
sudo tegrastats --interval "$INTERVAL_MS" --logfile "$TEGRA_LOG" &
TEGRA_PID=$!

if [[ "$BASELINE_S" != "0" ]]; then
  echo "Baseline: ${BASELINE_S}s (keep system idle for best attribution)..."
  sleep "$BASELINE_S"
fi

echo "Running workload in container '$CONTAINER'..."
set +e
docker exec -i "$CONTAINER" bash -lc "cd '$WORKDIR' && $CMD" 2>&1 | tee "$RUN_LOG"
WORKLOAD_EXIT=${PIPESTATUS[0]}
set -e

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

echo
exit "$WORKLOAD_EXIT"

