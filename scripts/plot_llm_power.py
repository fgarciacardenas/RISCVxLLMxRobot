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


def plot_power_per_test(segments: list[SegmentEnergy], outpath: str, rails: list[str]):
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

        def _safe_div(a: float, b: float) -> float:
            return a / b if b and b > 0 else float("nan")

        p_abs_w = [_safe_div(r.energy_j, r.dur_s) for r in rs_sorted]
        p_delta_w = [_safe_div(r.delta_energy_j, r.dur_s) for r in rs_sorted]

        ax.plot(xs, p_abs_w, color="#8aa1b1", linewidth=1.0, alpha=0.6, label="avg_power_W (abs)")
        ax.plot(xs, p_delta_w, color="#1f77b4", linewidth=1.5, marker="o", markersize=3, label="avg_delta_power_W (baseline-sub)")
        ax.set_ylabel("Power (W)")
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
    baseline_s: float | None,
    align_to_events: bool,
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

    intervals = read_llm_intervals(run_log)
    if not intervals:
        raise SystemExit("No LLMXROBOT_EVENT decode intervals found in workload log.")

    # Align tegrastats reconstructed timestamps to event clock.
    # tegrastats logs time only to 1s resolution, so the reconstructed timeline can
    # be shifted by up to ~1s. Apply a constant offset using the first decode start.
    shift_s = 0.0
    if align_to_events:
        first_start = min(s for s, _e, _m in intervals)
        i0 = int(round((first_start - ts[0]) / dt_s))
        i0 = max(0, min(i0, len(ts) - 1))
        shift_s = first_start - (ts[0] + i0 * dt_s)
        if abs(shift_s) > 2.0:
            shift_s = 0.0
        ts = [t + shift_s for t in ts]

    # Infer baseline window: default is samples strictly before the first decode start.
    first_start = min(s for s, _e, _m in intervals)
    if baseline_s is None:
        baseline_mask = [t < first_start for t in ts]
    else:
        baseline_end = ts[0] + float(baseline_s)
        baseline_mask = [t < baseline_end for t in ts]
    baseline_vals = [mw for mw, is_b in zip(p, baseline_mask) if is_b]
    baseline_mean = (sum(baseline_vals) / len(baseline_vals)) if baseline_vals else 0.0

    # Plot: absolute power (W) + delta power (W) in two subplots with the same markers.
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(14, 7), sharex=True)
    t0 = ts[0]
    x = [(t - t0) for t in ts]
    p_w = [mw / 1000.0 for mw in p]
    dp_w = [(mw - baseline_mean) / 1000.0 for mw in p]

    ax0.plot(x, p_w, linewidth=0.8, color="#333333", alpha=0.85, label=f"{rail} (W)")
    ax0.axhline(baseline_mean / 1000.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.9, label="baseline mean (W)")
    ax1.plot(x, dp_w, linewidth=0.9, color="#1f77b4", alpha=0.9, label=f"{rail} delta (W)")
    ax1.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)

    # Mark decode windows
    for i, (s, e, _meta) in enumerate(intervals):
        xs = s - t0
        xe = e - t0
        ax0.axvspan(xs, xe, color="#1f77b4", alpha=0.12)
        ax1.axvspan(xs, xe, color="#1f77b4", alpha=0.12)
        ax0.axvline(xs, color="#1f77b4", alpha=0.25, linewidth=0.8)
        ax0.axvline(xe, color="#1f77b4", alpha=0.25, linewidth=0.8)
        ax1.axvline(xs, color="#1f77b4", alpha=0.25, linewidth=0.8)
        ax1.axvline(xe, color="#1f77b4", alpha=0.25, linewidth=0.8)

    baseline_desc = "auto(pre-first-decode)" if baseline_s is None else f"{baseline_s:.0f}s"
    align_desc = f", align_shift_s={shift_s:.3f}" if align_to_events else ""
    ax0.set_title(f"Power ({rail}) with LLM decode windows (n={len(intervals)}), baseline={baseline_desc}{align_desc}")
    ax1.set_title("Delta power vs baseline (decode windows shaded)")
    ax1.set_xlabel("Time since start (s)")
    ax0.set_ylabel("Power (W)")
    ax1.set_ylabel("Delta power (W)")
    ax0.grid(True, alpha=0.25)
    ax1.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", fontsize=9)
    ax1.legend(loc="upper right", fontsize=9)
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
    ap.add_argument("--baseline-s", default="auto", help="Baseline duration in seconds, or 'auto' (default: auto = all samples before first decode).")
    ap.add_argument("--no-align", action="store_true", help="Disable auto alignment of tegrastats timestamps to decode event clock.")
    ap.add_argument("--rails", default="VIN_SYS_5V0,VDD_GPU_SOC,VDD_CPU_CV", help="Rails to plot for energy-per-test.")
    ap.add_argument("--trace-rail", default="VIN_SYS_5V0", help="Single rail to plot for the full-run trace.")
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

    power_plot = os.path.join(args.outdir, "power_per_test.png")
    plot_power_per_test(segments, outpath=power_plot, rails=rails)

    trace_plot = os.path.join(args.outdir, "power_trace_with_decode_windows.png")
    baseline_s: float | None
    if isinstance(args.baseline_s, str) and args.baseline_s.strip().lower() in ("auto", ""):
        baseline_s = None
    else:
        baseline_s = float(args.baseline_s)
    plot_power_with_markers(
        tegrastats_log=teg,
        run_log=run,
        outpath=trace_plot,
        rail=args.trace_rail,
        interval_ms=args.interval_ms,
        baseline_s=baseline_s,
        align_to_events=not args.no_align,
    )

    print("wrote:")
    print(f"  {energy_plot}")
    print(f"  {power_plot}")
    print(f"  {trace_plot}")


if __name__ == "__main__":
    raise SystemExit(main())
