#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys


def read_first_event_time(run_log: str, event: str, prefix: str = "LLMXROBOT_EVENT ") -> float | None:
    if not run_log:
        return None
    try:
        with open(run_log, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith(prefix):
                    continue
                try:
                    payload = json.loads(line[len(prefix) :].strip())
                except Exception:
                    continue
                if payload.get("event") != event:
                    continue
                t = payload.get("t_epoch_s")
                if isinstance(t, (int, float)):
                    return float(t)
    except FileNotFoundError:
        return None
    return None


def read_power_csv(path: str) -> tuple[list[float], list[float]]:
    ts = []
    p = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = float(row["t_epoch_s"])
                mw = float(row["power_mW"])
            except Exception:
                continue
            if math.isfinite(t) and math.isfinite(mw):
                ts.append(t)
                p.append(mw)
    xs = sorted(zip(ts, p), key=lambda x: x[0])
    return [t for t, _ in xs], [mw for _, mw in xs]


def compute_dt(ts: list[float]) -> list[float]:
    if len(ts) < 2:
        return [0.0 for _ in ts]
    dts = []
    for i in range(len(ts) - 1):
        d = ts[i + 1] - ts[i]
        if d <= 0:
            d = 0.0
        dts.append(d)
    sorted_d = sorted([d for d in dts if d > 0])
    last = sorted_d[len(sorted_d) // 2] if sorted_d else (dts[-1] if dts else 0.0)
    dts.append(last)
    return dts


def baseline_from_vals(vals: list[float], estimator: str) -> float:
    if not vals:
        return 0.0
    v = sorted(vals)
    if estimator == "min":
        return float(v[0])
    if estimator == "p10":
        return float(v[int((len(v) - 1) * 0.10)])
    if estimator == "p50":
        return float(v[int((len(v) - 1) * 0.50)])
    return float(sum(v) / len(v))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize NVIDIA GPU power CSV (avg/peak/energy; optional baseline subtraction).")
    ap.add_argument("--power-csv", required=True, help="CSV produced by scripts/log_nvidia_gpu_power.py")
    ap.add_argument("--run-log", default="", help="Optional workload log for workload_start marker inference.")
    ap.add_argument("--baseline-s", type=float, default=None, help="Baseline duration in seconds (optional).")
    ap.add_argument("--baseline-estimator", default="p10", choices=("mean", "p10", "p50", "min"))
    args = ap.parse_args()

    ts, p = read_power_csv(args.power_csv)
    if not ts:
        print("No samples found.", file=sys.stderr)
        return 2
    dts = compute_dt(ts)

    t0 = ts[0]
    workload_start_t = read_first_event_time(args.run_log, "workload_start")
    if workload_start_t is not None:
        baseline_end = workload_start_t
        baseline_mode = "workload_start_event"
    elif args.baseline_s is not None and args.baseline_s > 0:
        baseline_end = t0 + float(args.baseline_s)
        baseline_mode = "fixed_seconds"
    else:
        baseline_end = t0
        baseline_mode = "none"

    total_energy_mw_s = sum(mw * dt for mw, dt in zip(p, dts))
    total_time_s = sum(dts)
    total_avg_mw = (total_energy_mw_s / total_time_s) if total_time_s > 0 else 0.0
    total_peak_mw = max(p) if p else 0.0

    baseline_vals = [mw for t, mw in zip(ts, p) if t < baseline_end]
    baseline_mw = baseline_from_vals(baseline_vals, estimator=args.baseline_estimator) if baseline_end > t0 else 0.0

    run_energy_mw_s = sum(mw * dt for t, mw, dt in zip(ts, p, dts) if t >= baseline_end)
    run_time_s = sum(dt for t, dt in zip(ts, dts) if t >= baseline_end)
    run_avg_mw = (run_energy_mw_s / run_time_s) if run_time_s > 0 else 0.0
    run_peak_mw = max([mw for t, mw in zip(ts, p) if t >= baseline_end], default=0.0)
    run_delta_energy_j = (run_energy_mw_s - (baseline_mw * run_time_s)) / 1000.0

    print(
        f"GPU total: samples={len(ts)} dur_s={total_time_s:.3f} avg_mW={total_avg_mw:.1f} "
        f"peak_mW={total_peak_mw:.0f} energy_J={total_energy_mw_s/1000.0:.3f}"
    )
    if baseline_end > t0:
        print(
            f"GPU run:   dur_s={run_time_s:.3f} avg_mW={run_avg_mw:.1f} peak_mW={run_peak_mw:.0f} "
            f"energy_J={run_energy_mw_s/1000.0:.3f} baseline_mW={baseline_mw:.1f} "
            f"delta_energy_J={run_delta_energy_j:.3f} baseline_mode={baseline_mode} estimator={args.baseline_estimator}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

