#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import time

try:
    from tegrastats_utils import iter_tegrastats_samples, parse_tegrastats_time_to_epoch, weighted_mean_in_window
except ModuleNotFoundError:  # pragma: no cover
    from scripts.tegrastats_utils import iter_tegrastats_samples, parse_tegrastats_time_to_epoch, weighted_mean_in_window


def read_event_intervals(run_log: str, start_event: str, end_event: str, prefix: str = "LLMXROBOT_EVENT "):
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
                    # Attach end-of-interval fields too (e.g., token counts).
                    meta.update({f"end_{k}": v for k, v in payload.items() if k not in ("event",)})
                    intervals.append((start_t, end_t, meta))
                start_payload = None
                start_t = None
    return intervals, start_t, (start_payload or {})


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


def find_first_and_last_tegrastats_epoch(path: str) -> tuple[float | None, float | None]:
    first = None
    last = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            t = parse_tegrastats_time_to_epoch(line)
            if t is None:
                continue
            if first is None:
                first = float(t)
            last = float(t)
    return first, last


def integrate_samples(samples, intervals):
    """
    Integrate piecewise-constant samples (with per-sample dt) over [start,end] intervals.
    """
    per_interval_mw_s = [0.0 for _ in intervals]
    total_mw_s = 0.0
    max_mw = 0
    n = 0

    j = 0
    for ts, dt_s, p_mw in samples:
        n += 1
        max_mw = max(max_mw, int(p_mw))
        total_mw_s += float(p_mw) * float(dt_s)

        # advance interval pointer if needed
        while j < len(intervals) and ts >= intervals[j][1]:
            j += 1
        k = j
        end_ts = ts + dt_s
        while k < len(intervals):
            start, end, _meta = intervals[k]
            if end_ts <= start:
                break
            if ts >= end:
                k += 1
                continue
            overlap = max(0.0, min(end_ts, end) - max(ts, start))
            if overlap > 0:
                per_interval_mw_s[k] += float(p_mw) * overlap
            if end_ts <= end:
                break
            k += 1

    return {
        "samples": n,
        "max_mw": max_mw,
        "total_mw_s": total_mw_s,
        "interval_mw_s": per_interval_mw_s,
    }


def main():
    ap = argparse.ArgumentParser(description="Integrate tegrastats power over event-defined time intervals.")
    ap.add_argument("--tegrastats-log", required=True)
    ap.add_argument("--run-log", required=True)
    ap.add_argument("--interval-ms", type=int, required=True, help="Nominal tegrastats interval (kept for compatibility; analysis derives timing from timestamps).")
    ap.add_argument("--baseline-s", type=float, default=None, help="Baseline duration in seconds (preferred).")
    ap.add_argument("--baseline-samples", type=int, default=0, help="Fallback baseline: first N tegrastats samples.")
    ap.add_argument(
        "--baseline-estimator",
        default="mean",
        choices=("mean", "p10", "p50", "min"),
        help="How to estimate baseline power from the baseline window (default: mean).",
    )
    ap.add_argument("--rails", default="VIN_SYS_5V0", help="Comma-separated rails to integrate.")
    ap.add_argument("--start-event", default="llm_decode_start")
    ap.add_argument("--end-event", default="llm_decode_end")
    ap.add_argument("--out-csv", default="", help="Optional CSV output path.")
    args = ap.parse_args()

    rails = [r.strip() for r in args.rails.split(",") if r.strip()]
    intervals, open_start_t, open_meta = read_event_intervals(args.run_log, args.start_event, args.end_event)
    if open_start_t is not None:
        # If interrupted mid-decode, approximate the end as the last tegrastats timestamp (or "now").
        _first, last = find_first_and_last_tegrastats_epoch(args.tegrastats_log)
        end_t = last or time.time()
        if end_t >= open_start_t:
            intervals.append((open_start_t, end_t, open_meta))
    if not intervals:
        print("No event intervals found in run log; did you set LLMXROBOT_PROFILE_LLM=1?", file=sys.stderr)
        return 2

    # Determine baseline window.
    # Prefer a marker emitted by the profiler (so it stays idle-only even if model loads before first decode).
    workload_start_t = read_first_event_time(args.run_log, "workload_start")

    t0_wall, _t_last = find_first_and_last_tegrastats_epoch(args.tegrastats_log)
    if t0_wall is None:
        print("No tegrastats timestamps found.", file=sys.stderr)
        return 2

    first_decode_start = min(s for s, _e, _m in intervals)
    if workload_start_t is not None:
        baseline_end = workload_start_t
        baseline_mode = "workload_start_event"
    elif args.baseline_s is not None and args.baseline_s > 0:
        baseline_end = float(t0_wall) + float(args.baseline_s)
        baseline_mode = "fixed_seconds"
    elif args.baseline_samples > 0:
        baseline_end = None
        baseline_mode = "first_n_samples"
    else:
        baseline_end = first_decode_start
        baseline_mode = "pre_first_decode"

    def _pct(vals: list[int], p: float) -> float:
        if not vals:
            return 0.0
        v = sorted(vals)
        idx = int((len(v) - 1) * p)
        return float(v[idx])

    def _baseline_from_vals(vals: list[int]) -> float:
        if not vals:
            return 0.0
        if args.baseline_estimator == "min":
            return float(min(vals))
        if args.baseline_estimator == "p10":
            return _pct(vals, 0.10)
        if args.baseline_estimator == "p50":
            return _pct(vals, 0.50)
        return float(sum(vals) / len(vals))

    baseline_mean_mw: dict[str, float] = {}
    baseline_used_mw: dict[str, float] = {}
    baseline_n: dict[str, int] = {}

    for rail in rails:
        vals: list[int] = []
        if baseline_mode == "first_n_samples":
            for i, s in enumerate(iter_tegrastats_samples(args.tegrastats_log, rails=[rail])):
                if i >= args.baseline_samples:
                    break
                vals.append(int(s.rails_mw.get(rail, 0)))
        else:
            for s in iter_tegrastats_samples(args.tegrastats_log, rails=[rail]):
                if s.t_epoch_s >= baseline_end:
                    break
                vals.append(int(s.rails_mw.get(rail, 0)))

        baseline_n[rail] = len(vals)
        if baseline_mode == "first_n_samples":
            baseline_mean_mw[rail] = (float(sum(vals) / len(vals)) if vals else 0.0)
        else:
            baseline_mean_mw[rail] = float(
                weighted_mean_in_window(
                    iter_tegrastats_samples(args.tegrastats_log, rails=[rail]),
                    rail=rail,
                    t0=float(t0_wall),
                    t1=float(baseline_end),
                )
            )
        baseline_used_mw[rail] = _baseline_from_vals(vals)

    # Integrate per rail
    results: dict[str, dict] = {}
    for rail in rails:
        samples_iter = ((s.t_epoch_s, s.dt_s, s.rails_mw.get(rail, 0)) for s in iter_tegrastats_samples(args.tegrastats_log, rails=[rail]))
        results[rail] = integrate_samples(samples_iter, intervals)

    # Print summary and optional CSV
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
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
            ])
            for idx, (s, e, meta) in enumerate(intervals):
                dur = e - s
                prompt_chars = meta.get("prompt_chars")
                prompt_tokens = meta.get("end_prompt_tokens")
                completion_tokens = meta.get("end_completion_tokens")
                max_tokens = meta.get("max_tokens")
                for rail in rails:
                    e_j = results[rail]["interval_mw_s"][idx] / 1000.0
                    b_used = baseline_used_mw.get(rail, 0.0)
                    b_mean = baseline_mean_mw.get(rail, 0.0)
                    delta_j = (results[rail]["interval_mw_s"][idx] - (b_used * dur)) / 1000.0
                    w.writerow([
                        idx,
                        f"{s:.6f}",
                        f"{e:.6f}",
                        f"{dur:.6f}",
                        rail,
                        f"{e_j:.6f}",
                        f"{delta_j:.6f}",
                        f"{b_used:.3f}",
                        f"{b_mean:.3f}",
                        prompt_chars,
                        prompt_tokens,
                        completion_tokens,
                        max_tokens,
                        baseline_mode,
                        args.baseline_estimator,
                        baseline_n.get(rail, 0),
                    ])

    print(f"Intervals: {len(intervals)}  baseline_mode={baseline_mode}")
    for rail in rails:
        b = baseline_used_mw.get(rail, 0.0)
        energies = [mw_s / 1000.0 for mw_s in results[rail]["interval_mw_s"]]
        durs = [e - s for s, e, _m in intervals]
        delta = [((mw_s - (b * dur)) / 1000.0) for mw_s, dur in zip(results[rail]["interval_mw_s"], durs)]
        total_e = sum(energies)
        total_delta = sum(delta)
        avg_e = total_e / len(energies)
        avg_delta = total_delta / len(delta)
        print(
            f"{rail}: total_energy_J={total_e:.3f} avg_energy_J={avg_e:.3f} "
            f"total_delta_J={total_delta:.3f} avg_delta_J={avg_delta:.3f} baseline_mW={b:.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
