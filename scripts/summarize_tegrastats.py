#!/usr/bin/env python3
import argparse
import datetime as dt
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
        naive = dt.datetime.strptime(f"{d} {t}", "%m-%d-%Y %H:%M:%S")
        return time.mktime(naive.timetuple())
    except Exception:
        return None


def iter_samples(path: str, rail: str, dt_s: float):
    rail_re = re.compile(rf"{re.escape(rail)}\s+(\d+)mW\b")
    base_epoch = None
    idx = -1
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ts_wall = parse_tegrastats_time_to_epoch(line)
            if ts_wall is None:
                continue
            if base_epoch is None:
                base_epoch = ts_wall
                idx = 0
            else:
                idx += 1
            ts = float(base_epoch) + (idx * dt_s)
            m = rail_re.search(line)
            if not m:
                continue
            yield ts, int(m.group(1))


def summarize(path: str, rail: str, dt_s: float, baseline_samples: int):
    n = 0
    total_sum = 0.0
    total_max = 0

    b_n = 0
    b_sum = 0.0
    b_max = 0

    r_n = 0
    r_sum = 0.0
    r_max = 0

    for _ts, p_mw in iter_samples(path, rail, dt_s):
        n += 1
        total_sum += p_mw
        total_max = max(total_max, p_mw)
        if n <= baseline_samples:
            b_n += 1
            b_sum += p_mw
            b_max = max(b_max, p_mw)
        else:
            r_n += 1
            r_sum += p_mw
            r_max = max(r_max, p_mw)

    if n == 0:
        return None

    total = {
        "samples": n,
        "dur_s": n * dt_s,
        "avg_mw": total_sum / n,
        "peak_mw": total_max,
        "energy_j": (total_sum * dt_s) / 1000.0,
    }

    baseline_avg = (b_sum / b_n) if b_n > 0 else 0.0
    run = {
        "samples": r_n,
        "dur_s": r_n * dt_s,
        "avg_mw": (r_sum / r_n) if r_n > 0 else 0.0,
        "peak_mw": r_max,
        "energy_j": (r_sum * dt_s) / 1000.0,
        "baseline_mw": baseline_avg,
        "delta_energy_j": ((r_sum - (baseline_avg * r_n)) * dt_s) / 1000.0,
    }

    return total, run


def main():
    ap = argparse.ArgumentParser(description="Summarize tegrastats rail power (avg/peak/energy; optional baseline subtraction).")
    ap.add_argument("--tegrastats-log", required=True)
    ap.add_argument("--interval-ms", type=int, required=True)
    ap.add_argument("--baseline-samples", type=int, default=0)
    ap.add_argument("--rails", default="VIN_SYS_5V0,VDD_GPU_SOC,VDD_CPU_CV")
    args = ap.parse_args()

    dt_s = args.interval_ms / 1000.0
    rails = [r.strip() for r in args.rails.split(",") if r.strip()]

    missing = []
    for rail in rails:
        out = summarize(args.tegrastats_log, rail, dt_s, args.baseline_samples)
        if out is None:
            missing.append(rail)
            continue
        total, run = out
        print(
            f"{rail} total: samples={total['samples']} dur_s={total['dur_s']:.3f} "
            f"avg_mW={total['avg_mw']:.1f} peak_mW={total['peak_mw']} energy_J={total['energy_j']:.3f}"
        )
        if args.baseline_samples > 0:
            print(
                f"{rail} run:   samples={run['samples']} dur_s={run['dur_s']:.3f} "
                f"avg_mW={run['avg_mw']:.1f} peak_mW={run['peak_mw']} energy_J={run['energy_j']:.3f} "
                f"baseline_mW={run['baseline_mw']:.1f} delta_energy_J={run['delta_energy_j']:.3f}"
            )

    if missing:
        print(f"Warning: no samples found for rails: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
