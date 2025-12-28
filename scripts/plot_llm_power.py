#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import time
from dataclasses import dataclass

# Headless plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def iter_tegrastats_samples(path: str, rail: str, dt_s: float):
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


def read_llm_intervals(run_log: str, prefix: str = "LLMXROBOT_EVENT "):
    """
    Returns list of (start_epoch_s, end_epoch_s, meta) for llm_decode_start/end pairs.
    Meta includes prompt token counts etc when present.
    """
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
            if ev == "llm_decode_start":
                start_payload = payload
                start_t = float(t)
            elif ev == "llm_decode_end" and start_t is not None:
                end_t = float(t)
                if end_t >= start_t:
                    meta = dict(start_payload or {})
                    meta.update({f"end_{k}": v for k, v in payload.items() if k != "event"})
                    intervals.append((start_t, end_t, meta))
                start_payload = None
                start_t = None
    return intervals


@dataclass(frozen=True)
class SegmentEnergy:
    idx: int
    rail: str
    dur_s: float
    energy_j: float
    delta_energy_j: float


def read_segments_csv(path: str) -> list[SegmentEnergy]:
    out: list[SegmentEnergy] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for rec in r:
            try:
                idx = int(float(rec["idx"]))
                rail = rec["rail"].strip()
                dur = float(rec["dur_s"])
                e = float(rec["energy_J"])
                de = float(rec["delta_energy_J"])
            except Exception:
                continue
            if math.isfinite(dur) and math.isfinite(e) and math.isfinite(de):
                out.append(SegmentEnergy(idx=idx, rail=rail, dur_s=dur, energy_j=e, delta_energy_j=de))
    return out


def plot_energy_per_test(segments: list[SegmentEnergy], outpath: str, rails: list[str]):
    by_rail: dict[str, list[SegmentEnergy]] = {}
    for s in segments:
        if rails and s.rail not in rails:
            continue
        by_rail.setdefault(s.rail, []).append(s)

    if not by_rail:
        raise SystemExit("No segment rows found for requested rails.")

    nrows = len(by_rail)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, (rail, rs) in zip(axes, sorted(by_rail.items())):
        rs_sorted = sorted(rs, key=lambda x: x.idx)
        xs = [r.idx for r in rs_sorted]
        y_abs = [r.energy_j for r in rs_sorted]
        y_delta = [r.delta_energy_j for r in rs_sorted]

        ax.plot(xs, y_abs, color="#8aa1b1", linewidth=1.0, alpha=0.6, label="energy_J (abs)")
        ax.plot(xs, y_delta, color="#1f77b4", linewidth=1.5, marker="o", markersize=3, label="delta_energy_J (baseline-sub)")
        ax.set_ylabel("Energy (J)")
        ax.set_title(rail)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Decode window idx")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_power_with_markers(
    tegrastats_log: str,
    run_log: str,
    outpath: str,
    rail: str,
    interval_ms: int,
    baseline_s: float,
):
    dt_s = interval_ms / 1000.0
    # Load power samples
    ts = []
    p = []
    for t, mw in iter_tegrastats_samples(tegrastats_log, rail=rail, dt_s=dt_s):
        ts.append(t)
        p.append(mw)
    if not ts:
        raise SystemExit(f"No tegrastats samples found for rail {rail}")

    # Infer baseline window (first baseline_s seconds worth of samples)
    baseline_samples = int(round(baseline_s / dt_s)) if baseline_s > 0 else 0
    baseline_mean = sum(p[:baseline_samples]) / max(1, min(len(p), baseline_samples)) if baseline_samples else 0.0

    intervals = read_llm_intervals(run_log)
    if not intervals:
        raise SystemExit("No LLMXROBOT_EVENT decode intervals found in workload log.")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    t0 = ts[0]
    x = [(t - t0) for t in ts]
    ax.plot(x, p, linewidth=0.8, color="#333333", alpha=0.8, label=f"{rail} mW")
    if baseline_samples:
        ax.axhline(baseline_mean, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8, label=f"baseline mean ({baseline_s:.0f}s)")

    # Mark decode windows
    for i, (s, e, _meta) in enumerate(intervals):
        xs = s - t0
        xe = e - t0
        ax.axvspan(xs, xe, color="#1f77b4", alpha=0.12)
        # light start/end lines
        ax.axvline(xs, color="#1f77b4", alpha=0.25, linewidth=0.8)
        ax.axvline(xe, color="#1f77b4", alpha=0.25, linewidth=0.8)

    ax.set_title(f"Power trace ({rail}) with LLM decode windows (n={len(intervals)})")
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Power (mW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot LLM segment energy + tegrastats power trace with decode markers.")
    ap.add_argument("--logdir", default="", help="Directory containing tegrastats_*.log/workload_*.log/llm_segments_*.csv")
    ap.add_argument("--segments-csv", default="", help="Path to llm_segments_*.csv (optional if --logdir is set).")
    ap.add_argument("--tegrastats-log", default="", help="Path to tegrastats_*.log (optional if --logdir is set).")
    ap.add_argument("--run-log", default="", help="Path to workload_*.log (optional if --logdir is set).")
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--baseline-s", type=float, default=30.0)
    ap.add_argument("--rails", default="VIN_SYS_5V0,VDD_GPU_SOC,VDD_CPU_CV", help="Rails to plot for energy-per-test.")
    ap.add_argument("--trace-rail", default="VDD_GPU_SOC", help="Single rail to plot for the full-run trace.")
    ap.add_argument("--outdir", default="plots", help="Output directory for images.")
    args = ap.parse_args()

    logdir = args.logdir.strip()
    seg = args.segments_csv.strip()
    teg = args.tegrastats_log.strip()
    run = args.run_log.strip()

    if logdir:
        # Prefer most recent matching set.
        def _pick_latest(prefix: str, suffix: str):
            cands = [f for f in os.listdir(logdir) if f.startswith(prefix) and f.endswith(suffix)]
            return os.path.join(logdir, sorted(cands)[-1]) if cands else ""

        if not seg:
            seg = _pick_latest("llm_segments_", ".csv")
        if not teg:
            teg = _pick_latest("tegrastats_", ".log")
        if not run:
            run = _pick_latest("workload_", ".log")

    if not seg or not os.path.exists(seg):
        raise SystemExit("Missing --segments-csv (or no llm_segments_*.csv found in --logdir).")
    if not teg or not os.path.exists(teg):
        raise SystemExit("Missing --tegrastats-log (or no tegrastats_*.log found in --logdir).")
    if not run or not os.path.exists(run):
        raise SystemExit("Missing --run-log (or no workload_*.log found in --logdir).")

    rails = [r.strip() for r in args.rails.split(",") if r.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    segments = read_segments_csv(seg)
    energy_plot = os.path.join(args.outdir, "energy_per_test.png")
    plot_energy_per_test(segments, outpath=energy_plot, rails=rails)

    trace_plot = os.path.join(args.outdir, "power_trace_with_decode_windows.png")
    plot_power_with_markers(
        tegrastats_log=teg,
        run_log=run,
        outpath=trace_plot,
        rail=args.trace_rail,
        interval_ms=args.interval_ms,
        baseline_s=args.baseline_s,
    )

    print("wrote:")
    print(f"  {energy_plot}")
    print(f"  {trace_plot}")


if __name__ == "__main__":
    raise SystemExit(main())

