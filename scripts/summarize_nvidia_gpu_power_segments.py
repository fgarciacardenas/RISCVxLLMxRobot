#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass


def read_llm_intervals(run_log: str, start_event: str, end_event: str, prefix: str = "LLMXROBOT_EVENT "):
    intervals: list[tuple[float, float, dict]] = []
    start_payload: dict | None = None
    start_t: float | None = None
    with open(run_log, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(prefix):
                continue
            try:
                payload = json.loads(line[len(prefix) :].strip())
            except Exception:
                continue
            ev = payload.get("event")
            t = payload.get("t_epoch_s")
            if not isinstance(t, (int, float)):
                continue
            if ev == start_event:
                start_payload = payload
                start_t = float(t)
            elif ev == end_event and start_t is not None:
                end_t = float(t)
                if end_t >= start_t:
                    meta = dict(start_payload or {})
                    meta.update({f"end_{k}": v for k, v in payload.items() if k != "event"})
                    intervals.append((start_t, end_t, meta))
                start_payload = None
                start_t = None
    return intervals


def read_first_event_time(run_log: str, event: str, prefix: str = "LLMXROBOT_EVENT ") -> float | None:
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
    return None


@dataclass(frozen=True)
class PowerSample:
    t: float
    p_mw: float


def read_power_csv(path: str) -> list[PowerSample]:
    out: list[PowerSample] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = float(row["t_epoch_s"])
                p = float(row["power_mW"])
            except Exception:
                continue
            if math.isfinite(t) and math.isfinite(p):
                out.append(PowerSample(t=t, p_mw=p))
    out.sort(key=lambda s: s.t)
    return out


def compute_dt(samples: list[PowerSample]) -> list[float]:
    if len(samples) < 2:
        return [0.0 for _ in samples]
    dts = []
    for i in range(len(samples) - 1):
        d = samples[i + 1].t - samples[i].t
        if d <= 0:
            d = 0.0
        dts.append(d)
    # last dt: reuse median of previous (robust-ish)
    sorted_d = sorted([d for d in dts if d > 0])
    last = sorted_d[len(sorted_d) // 2] if sorted_d else (dts[-1] if dts else 0.0)
    dts.append(last)
    return dts


def integrate_over_intervals(samples: list[PowerSample], dts: list[float], intervals: list[tuple[float, float, dict]]):
    per_interval_mw_s = [0.0 for _ in intervals]
    if not samples or not intervals:
        return per_interval_mw_s

    j = 0
    for s, dt_s in zip(samples, dts):
        a = s.t
        b = s.t + dt_s
        if dt_s <= 0:
            continue
        while j < len(intervals) and a >= intervals[j][1]:
            j += 1
        k = j
        while k < len(intervals):
            start, end, _meta = intervals[k]
            if b <= start:
                break
            if a >= end:
                k += 1
                continue
            overlap = max(0.0, min(b, end) - max(a, start))
            if overlap > 0:
                per_interval_mw_s[k] += float(s.p_mw) * overlap
            if b <= end:
                break
            k += 1
    return per_interval_mw_s


def baseline_from_window(samples: list[PowerSample], t0: float, t1: float, estimator: str) -> tuple[float, float, int]:
    """
    Returns (baseline_used_mW, baseline_mean_mW, n_samples_used).
    baseline_mean_mW is time-weighted. baseline_used_mW applies the estimator to sample values.
    """
    if t1 <= t0:
        return 0.0, 0.0, 0
    window = [s for s in samples if t0 <= s.t < t1]
    if not window:
        return 0.0, 0.0, 0

    # Unweighted estimator (p10/p50/min) on sample values.
    vals = sorted([float(s.p_mw) for s in window])
    if estimator == "min":
        used = vals[0]
    elif estimator == "p10":
        used = vals[int((len(vals) - 1) * 0.10)]
    elif estimator == "p50":
        used = vals[int((len(vals) - 1) * 0.50)]
    else:
        used = sum(vals) / len(vals)

    # Time-weighted mean.
    mean = used
    if len(window) >= 2:
        # Use adjacent timestamps inside the window.
        num = 0.0
        den = 0.0
        for i in range(len(window) - 1):
            a = window[i].t
            b = min(window[i + 1].t, t1)
            if b <= a:
                continue
            dt_s = b - a
            num += window[i].p_mw * dt_s
            den += dt_s
        if den > 0:
            mean = num / den

    return float(used), float(mean), len(window)


def main() -> int:
    ap = argparse.ArgumentParser(description="Integrate NVIDIA GPU power samples over LLM decode event intervals.")
    ap.add_argument("--power-csv", required=True, help="CSV produced by scripts/log_nvidia_gpu_power.py")
    ap.add_argument("--run-log", required=True, help="workload_*.log containing LLMXROBOT_EVENT markers")
    ap.add_argument("--out-csv", required=True, help="Output llm_segments_*.csv path")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index for rail label (default: 0)")
    ap.add_argument("--start-event", default="llm_decode_start")
    ap.add_argument("--end-event", default="llm_decode_end")
    ap.add_argument("--baseline-s", type=float, default=None, help="Baseline duration in seconds (optional).")
    ap.add_argument("--baseline-estimator", default="p10", choices=("mean", "p10", "p50", "min"))
    args = ap.parse_args()

    samples = read_power_csv(args.power_csv)
    if not samples:
        print("No power samples found.", file=sys.stderr)
        return 2
    dts = compute_dt(samples)

    intervals = read_llm_intervals(args.run_log, args.start_event, args.end_event)
    if not intervals:
        print("No decode intervals found in run log.", file=sys.stderr)
        return 2

    t0 = samples[0].t
    workload_start_t = read_first_event_time(args.run_log, "workload_start")
    first_decode_start = min(s for s, _e, _m in intervals)

    if workload_start_t is not None:
        baseline_end = workload_start_t
        baseline_mode = "workload_start_event"
    elif args.baseline_s is not None and args.baseline_s > 0:
        baseline_end = t0 + float(args.baseline_s)
        baseline_mode = "fixed_seconds"
    else:
        baseline_end = first_decode_start
        baseline_mode = "pre_first_decode"

    b_used, b_mean, b_n = baseline_from_window(samples, t0=t0, t1=baseline_end, estimator=args.baseline_estimator)

    per_interval_mw_s = integrate_over_intervals(samples, dts, intervals)
    rail = f"GPU{int(args.gpu)}"

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "start_epoch_s",
                "end_epoch_s",
                "dur_s",
                "rail",
                "energy_J",
                "delta_energy_J",
                "baseline_mW",
                "baseline_mean_mW",
                "prompt_chars",
                "prompt_tokens",
                "completion_tokens",
                "max_tokens",
                "baseline_mode",
                "baseline_estimator",
                "baseline_n",
            ]
        )
        for idx, ((s, e, meta), mw_s) in enumerate(zip(intervals, per_interval_mw_s)):
            dur = float(e - s)
            energy_j = float(mw_s) / 1000.0
            delta_j = (float(mw_s) - (b_used * dur)) / 1000.0
            w.writerow(
                [
                    idx,
                    f"{float(s):.6f}",
                    f"{float(e):.6f}",
                    f"{dur:.6f}",
                    rail,
                    f"{energy_j:.6f}",
                    f"{delta_j:.6f}",
                    f"{b_used:.3f}",
                    f"{b_mean:.3f}",
                    meta.get("prompt_chars"),
                    meta.get("end_prompt_tokens"),
                    meta.get("end_completion_tokens"),
                    meta.get("max_tokens"),
                    baseline_mode,
                    args.baseline_estimator,
                    b_n,
                ]
            )

    print(f"Intervals: {len(intervals)} baseline_mode={baseline_mode} baseline_mW={b_used:.1f} (mean_mW={b_mean:.1f}, n={b_n}, estimator={args.baseline_estimator})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

