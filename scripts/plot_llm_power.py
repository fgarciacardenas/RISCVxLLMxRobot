#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass

# Headless plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tegrastats_utils import iter_tegrastats_samples
except ModuleNotFoundError:  # pragma: no cover
    from scripts.tegrastats_utils import iter_tegrastats_samples


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
    baseline_mw: float | None


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
                bmw = float(rec["baseline_mW"]) if rec.get("baseline_mW") not in (None, "", "None") else None
            except Exception:
                continue
            if math.isfinite(dur) and math.isfinite(e) and math.isfinite(de):
                out.append(SegmentEnergy(idx=idx, rail=rail, dur_s=dur, energy_j=e, delta_energy_j=de, baseline_mw=bmw))
    return out


def read_segments_csv_meta(path: str) -> dict[str, str]:
    """
    Best-effort: extract run-wide metadata columns (baseline_mode/estimator) from the first row.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            r = csv.DictReader(f)
            first = next(r, None)
            if not first:
                return {}
            meta = {}
            for k in ("baseline_mode", "baseline_estimator", "baseline_n"):
                if k in first and first[k] not in (None, ""):
                    meta[k] = str(first[k])
            return meta
    except Exception:
        return {}


def plot_energy_per_test(segments: list[SegmentEnergy], outpath: str, rails: list[str], clamp_delta: bool):
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
        y_delta = [max(0.0, r.delta_energy_j) if clamp_delta else r.delta_energy_j for r in rs_sorted]

        ax.plot(xs, y_abs, color="#8aa1b1", linewidth=1.0, alpha=0.6, label="energy_J (abs)")
        ax.plot(xs, y_delta, color="#1f77b4", linewidth=1.5, marker="o", markersize=3, label="delta_energy_J (baseline-sub)")
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylabel("Energy (J)")
        ax.set_title(rail)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Decode window idx")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    fig.savefig(os.path.splitext(outpath)[0] + ".pdf")
    plt.close(fig)


def plot_power_per_test(segments: list[SegmentEnergy], outpath: str, rails: list[str], clamp_delta: bool):
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
        p_delta_w = [_safe_div(max(0.0, r.delta_energy_j) if clamp_delta else r.delta_energy_j, r.dur_s) for r in rs_sorted]

        ax.plot(xs, p_abs_w, color="#8aa1b1", linewidth=1.0, alpha=0.6, label="avg_power_W (abs)")
        ax.plot(xs, p_delta_w, color="#1f77b4", linewidth=1.5, marker="o", markersize=3, label="avg_delta_power_W (baseline-sub)")
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylabel("Power (W)")
        ax.set_title(rail)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Decode window idx")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    fig.savefig(os.path.splitext(outpath)[0] + ".pdf")
    plt.close(fig)


def plot_power_with_markers(
    tegrastats_log: str,
    run_log: str,
    outpath: str,
    rail: str,
    interval_ms: int,
    baseline_s: float | None,
    baseline_mw_override: float | None,
    align_to_events: bool,
):
    # Load power samples
    ts = []
    dts = []
    p = []
    for s in iter_tegrastats_samples(tegrastats_log, rails=[rail]):
        ts.append(float(s.t_epoch_s))
        dts.append(float(s.dt_s))
        p.append(int(s.rails_mw.get(rail, 0)))
    if not ts:
        raise SystemExit(f"No tegrastats samples found for rail {rail}")

    intervals = read_llm_intervals(run_log)
    if not intervals:
        raise SystemExit("No LLMXROBOT_EVENT decode intervals found in workload log.")

    # Align tegrastats timestamps to event clock.
    # tegrastats timestamps are only 1s resolution; apply a small constant offset
    # by snapping the nearest tegrastats sample to the first decode start.
    shift_s = 0.0
    if align_to_events:
        first_start = min(s for s, _e, _m in intervals)
        i0 = min(range(len(ts)), key=lambda i: abs(ts[i] - first_start))
        shift_s = float(first_start) - float(ts[i0])
        if abs(shift_s) > 2.0:
            shift_s = 0.0
        ts = [t + shift_s for t in ts]

    # Infer baseline window: default is samples strictly before the first decode start.
    first_start = min(s for s, _e, _m in intervals)
    if baseline_mw_override is not None and math.isfinite(baseline_mw_override):
        baseline_mean = float(baseline_mw_override)
        baseline_mode = "from_llm_segments_csv"
        baseline_samples = 0
    else:
        if baseline_s is None:
            baseline_end = float(first_start)
            baseline_mode = "auto_pre_first_decode"
        else:
            baseline_end = float(ts[0]) + float(baseline_s)
            baseline_mode = "fixed_seconds"

        num = 0.0
        den = 0.0
        used = 0
        for t, dt_s, mw in zip(ts, dts, p):
            a = t
            b = t + dt_s
            overlap = max(0.0, min(b, baseline_end) - max(a, ts[0]))
            if overlap <= 0:
                continue
            num += float(mw) * overlap
            den += overlap
            used += 1
        baseline_mean = (num / den) if den > 0 else 0.0
        baseline_samples = used

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

    if baseline_mode == "from_llm_segments_csv":
        baseline_desc = "from llm_segments CSV"
    elif baseline_s is None:
        baseline_desc = "auto(pre-first-decode)"
    else:
        baseline_desc = f"{baseline_s:.0f}s"
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
    fig.savefig(os.path.splitext(outpath)[0] + ".pdf")
    plt.close(fig)
    return {
        "align_shift_s": shift_s,
        "baseline_mean_mW": baseline_mean,
        "baseline_samples": baseline_samples,
        "baseline_mode": baseline_mode,
    }


def plot_delta_power_only_with_markers(
    tegrastats_log: str,
    run_log: str,
    outpath: str,
    rails: list[str],
    segments: list[SegmentEnergy],
    align_to_events: bool,
):
    intervals = read_llm_intervals(run_log)
    if not intervals:
        raise SystemExit("No LLMXROBOT_EVENT decode intervals found in workload log.")

    # Baselines from llm_segments CSV (so deltas match segment integration).
    baseline_by_rail_mw: dict[str, float] = {}
    for rail in rails:
        bs = [s.baseline_mw for s in segments if s.rail == rail and s.baseline_mw is not None and math.isfinite(s.baseline_mw)]
        baseline_by_rail_mw[rail] = float(sum(bs) / len(bs)) if bs else 0.0

    # Load tegrastats stream containing all rails (same timestamps for all rails).
    ts: list[float] = []
    p_mw_by_rail: dict[str, list[float]] = {r: [] for r in rails}
    for s in iter_tegrastats_samples(tegrastats_log, rails=rails):
        ts.append(float(s.t_epoch_s))
        for r in rails:
            p_mw_by_rail[r].append(float(s.rails_mw.get(r, 0)))
    if not ts:
        raise SystemExit("No tegrastats samples found.")

    shift_s = 0.0
    if align_to_events:
        first_start = min(s for s, _e, _m in intervals)
        i0 = min(range(len(ts)), key=lambda i: abs(ts[i] - first_start))
        shift_s = float(first_start) - float(ts[i0])
        if abs(shift_s) > 2.0:
            shift_s = 0.0
        ts = [t + shift_s for t in ts]

    t0 = ts[0]
    x = [t - t0 for t in ts]

    name_map = {"VDD_CPU_CV": "CPU", "VDD_GPU_SOC": "GPU", "VIN_SYS_5V0": "VIN"}
    baseline_desc = ", ".join([f"{name_map.get(r, r)}={baseline_by_rail_mw[r]/1000.0:.3f} W" for r in rails])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 4.3), sharex=True)
    for r in rails:
        b = baseline_by_rail_mw.get(r, 0.0)
        dp_w = [((mw - b) / 1000.0) for mw in p_mw_by_rail[r]]
        ax.plot(x, dp_w, linewidth=0.9, alpha=0.95, label=name_map.get(r, r))

    for (s, e, _meta) in intervals:
        xs = s - t0
        xe = e - t0
        ax.axvspan(xs, xe, color="#1f77b4", alpha=0.10)

    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"Jetson power plot. Baseline substracted: {baseline_desc}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, ncol=min(3, len(rails)))
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    fig.savefig(os.path.splitext(outpath)[0] + ".pdf")
    plt.close(fig)
    return {"align_shift_s": shift_s, "baseline_by_rail_mW": baseline_by_rail_mw}


def print_segment_baselines(segments: list[SegmentEnergy], rails: list[str]):
    by_rail: dict[str, list[float]] = {}
    neg_by_rail: dict[str, int] = {}
    for s in segments:
        if rails and s.rail not in rails:
            continue
        if s.baseline_mw is not None and math.isfinite(s.baseline_mw):
            by_rail.setdefault(s.rail, []).append(s.baseline_mw)
        if s.delta_energy_j < 0:
            neg_by_rail[s.rail] = neg_by_rail.get(s.rail, 0) + 1
    if by_rail:
        print("Baselines from llm_segments CSV (baseline_mW):")
        for rail, vals in sorted(by_rail.items()):
            mean = sum(vals) / len(vals)
            print(f"  {rail}: mean={mean:.1f} mW  min={min(vals):.1f}  max={max(vals):.1f}  n={len(vals)}")
    if neg_by_rail:
        print("Note: negative delta_energy_J rows (decode window below baseline mean) per rail:")
        for rail, n in sorted(neg_by_rail.items()):
            print(f"  {rail}: {n}")


def main():
    ap = argparse.ArgumentParser(description="Plot LLM segment energy + tegrastats power trace with decode markers.")
    ap.add_argument("--logdir", default="", help="Directory containing tegrastats_*.log/workload_*.log/llm_segments_*.csv")
    ap.add_argument("--segments-csv", default="", help="Path to llm_segments_*.csv (optional if --logdir is set).")
    ap.add_argument("--tegrastats-log", default="", help="Path to tegrastats_*.log (optional if --logdir is set).")
    ap.add_argument("--run-log", default="", help="Path to workload_*.log (optional if --logdir is set).")
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--baseline-s", default="auto", help="Baseline duration in seconds, or 'auto' (default: auto = all samples before first decode).")
    ap.add_argument("--no-align", action="store_true", help="Disable auto alignment of tegrastats timestamps to decode event clock.")
    ap.add_argument("--clamp-delta", action="store_true", help="Clamp delta energy/power at 0 in per-test plots (visualization only).")
    ap.add_argument("--rails", default="VIN_SYS_5V0,VDD_GPU_SOC,VDD_CPU_CV", help="Rails to plot for energy-per-test.")
    ap.add_argument("--trace-rail", default="VDD_GPU_SOC", help="Single rail to plot for the full-run trace.")
    ap.add_argument("--delta-only-rails", default="VDD_CPU_CV,VDD_GPU_SOC,VIN_SYS_5V0", help="Rails to plot for the delta-only trace.")
    ap.add_argument("--outdir", default="plots", help="Output directory for images.")
    args = ap.parse_args()

    logdir = args.logdir.strip()
    seg = args.segments_csv.strip()
    teg = args.tegrastats_log.strip()
    run = args.run_log.strip()

    if logdir:
        # Prefer most recent matching set.
        def _pick_latest(prefix: str, suffix: str):
            cands = [os.path.join(logdir, f) for f in os.listdir(logdir) if f.startswith(prefix) and f.endswith(suffix)]
            if not cands:
                return ""
            # Pick by modification time (lexicographic order can be misleading once you start adding suffixes).
            return max(cands, key=lambda p: os.path.getmtime(p))

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
    seg_meta = read_segments_csv_meta(seg)
    print("Using logs:")
    print(f"  segments:   {seg}")
    if seg_meta:
        print("  segments_meta: " + ", ".join([f"{k}={v}" for k, v in sorted(seg_meta.items())]))
    print(f"  tegrastats:  {teg}")
    print(f"  workload:    {run}")

    energy_plot = os.path.join(args.outdir, "energy_per_test.png")
    plot_energy_per_test(segments, outpath=energy_plot, rails=rails, clamp_delta=args.clamp_delta)

    power_plot = os.path.join(args.outdir, "power_per_test.png")
    plot_power_per_test(segments, outpath=power_plot, rails=rails, clamp_delta=args.clamp_delta)

    trace_plot = os.path.join(args.outdir, "power_trace_with_decode_windows.png")
    baseline_s: float | None
    if isinstance(args.baseline_s, str) and args.baseline_s.strip().lower() in ("auto", ""):
        baseline_s = None
    else:
        baseline_s = float(args.baseline_s)
    baseline_override_mw: float | None = None
    if segments:
        bs = [s.baseline_mw for s in segments if s.rail == args.trace_rail and s.baseline_mw is not None and math.isfinite(s.baseline_mw)]
        if bs:
            baseline_override_mw = sum(bs) / len(bs)

    trace_meta = plot_power_with_markers(
        tegrastats_log=teg,
        run_log=run,
        outpath=trace_plot,
        rail=args.trace_rail,
        interval_ms=args.interval_ms,
        baseline_s=baseline_s,
        baseline_mw_override=baseline_override_mw,
        align_to_events=not args.no_align,
    )

    delta_rails = [r.strip() for r in args.delta_only_rails.split(",") if r.strip()]
    delta_plot = os.path.join(args.outdir, "delta_power_trace_only.png")
    delta_meta = plot_delta_power_only_with_markers(
        tegrastats_log=teg,
        run_log=run,
        outpath=delta_plot,
        rails=delta_rails,
        segments=segments,
        align_to_events=not args.no_align,
    )

    print_segment_baselines(segments, rails=rails)
    print(
        f"Trace baseline ({args.trace_rail}): mean={trace_meta['baseline_mean_mW']/1000.0:.3f} W "
        f"(samples={trace_meta['baseline_samples']}, mode={trace_meta['baseline_mode']}, align_shift_s={trace_meta['align_shift_s']:.3f})"
    )
    print(
        "Delta-only trace baselines: "
        + ", ".join([f"{k}={v/1000.0:.3f} W" for k, v in sorted(delta_meta["baseline_by_rail_mW"].items())])
    )

    print("wrote:")
    print(f"  {energy_plot}")
    print(f"  {power_plot}")
    print(f"  {trace_plot}")
    print(f"  {delta_plot}")


if __name__ == "__main__":
    raise SystemExit(main())
