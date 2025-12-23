#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import re
import sys
import time


TEGRA_TS_RE = re.compile(r"^\s*(\d{2}-\d{2}-\d{4})\s+(\d{2}:\d{2}:\d{2})\s+")


def parse_tegrastats_time_to_epoch(line: str) -> float | None:
    m = TEGRA_TS_RE.match(line)
    if not m:
        return None
    d, t = m.group(1), m.group(2)
    try:
        # tegrastats prints local time without tz; interpret as local.
        naive = dt.datetime.strptime(f"{d} {t}", "%m-%d-%Y %H:%M:%S")
        return time.mktime(naive.timetuple())
    except Exception:
        return None


def iter_tegrastats_samples(path: str, rail: str):
    rail_re = re.compile(rf"{re.escape(rail)}\s+(\d+)mW\b")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ts = parse_tegrastats_time_to_epoch(line)
            if ts is None:
                continue
            m = rail_re.search(line)
            if not m:
                continue
            yield ts, int(m.group(1))


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
                    intervals.append((start_t, end_t, start_payload or {}))
                start_payload = None
                start_t = None
    return intervals, start_t, (start_payload or {})


def last_tegrastats_epoch(path: str) -> float | None:
    last = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ts = parse_tegrastats_time_to_epoch(line)
            if ts is not None:
                last = ts
    return last


def integrate(samples, intervals, dt_s: float):
    """
    Integrate piecewise-constant samples over [start,end] intervals.
    Each sample at timestamp ts is assumed to represent [ts, ts+dt_s).
    """
    per_interval_mw_s = [0.0 for _ in intervals]
    total_mw_s = 0.0
    max_mw = 0
    n = 0

    j = 0
    for ts, p_mw in samples:
        n += 1
        max_mw = max(max_mw, p_mw)
        total_mw_s += p_mw * dt_s

        # advance interval pointer if needed
        while j < len(intervals) and ts >= intervals[j][1]:
            j += 1
        k = j
        while k < len(intervals) and ts < intervals[k][0]:
            break
        if k < len(intervals):
            start, end, _meta = intervals[k]
            if start <= ts < end:
                per_interval_mw_s[k] += p_mw * dt_s

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
    ap.add_argument("--interval-ms", type=int, required=True)
    ap.add_argument("--baseline-samples", type=int, default=0, help="Number of initial tegrastats samples to treat as baseline window.")
    ap.add_argument("--rails", default="VIN_SYS_5V0", help="Comma-separated rails to integrate.")
    ap.add_argument("--start-event", default="llm_decode_start")
    ap.add_argument("--end-event", default="llm_decode_end")
    ap.add_argument("--out-csv", default="", help="Optional CSV output path.")
    args = ap.parse_args()

    dt_s = args.interval_ms / 1000.0
    rails = [r.strip() for r in args.rails.split(",") if r.strip()]
    intervals, open_start_t, open_meta = read_event_intervals(args.run_log, args.start_event, args.end_event)
    if open_start_t is not None:
        # If interrupted mid-decode, approximate the end as the last tegrastats timestamp (or "now").
        end_t = last_tegrastats_epoch(args.tegrastats_log) or time.time()
        if end_t >= open_start_t:
            intervals.append((open_start_t, end_t, open_meta))
    if not intervals:
        print("No event intervals found in run log; did you set LLMXROBOT_PROFILE_LLM=1?", file=sys.stderr)
        return 2

    # Baseline: compute mean over first N samples for each rail (if provided)
    baseline_mean_mw: dict[str, float] = {}
    if args.baseline_samples > 0:
        for rail in rails:
            vals = []
            for i, (_ts, p_mw) in enumerate(iter_tegrastats_samples(args.tegrastats_log, rail)):
                if i >= args.baseline_samples:
                    break
                vals.append(p_mw)
            baseline_mean_mw[rail] = (sum(vals) / len(vals)) if vals else 0.0
    else:
        for rail in rails:
            baseline_mean_mw[rail] = 0.0

    # Integrate per rail
    results: dict[str, dict] = {}
    for rail in rails:
        samples_iter = iter_tegrastats_samples(args.tegrastats_log, rail)
        results[rail] = integrate(samples_iter, intervals, dt_s)

    # Print summary and optional CSV
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "start_epoch_s", "end_epoch_s", "dur_s", "rail", "energy_J", "delta_energy_J", "baseline_mW"])
            for idx, (s, e, _meta) in enumerate(intervals):
                dur = e - s
                for rail in rails:
                    e_j = results[rail]["interval_mw_s"][idx] / 1000.0
                    b = baseline_mean_mw.get(rail, 0.0)
                    delta_j = (results[rail]["interval_mw_s"][idx] - (b * dur)) / 1000.0
                    w.writerow([idx, f"{s:.6f}", f"{e:.6f}", f"{dur:.6f}", rail, f"{e_j:.6f}", f"{delta_j:.6f}", f"{b:.3f}"])

    print(f"Intervals: {len(intervals)}  dt_s={dt_s:.3f}")
    for rail in rails:
        b = baseline_mean_mw.get(rail, 0.0)
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
