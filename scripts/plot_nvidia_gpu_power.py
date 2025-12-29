#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_llm_intervals(run_log: str, prefix: str = "LLMXROBOT_EVENT "):
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


def read_power_csv(path: str):
    ts = []
    p_mw = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ts.append(float(row["t_epoch_s"]))
                p_mw.append(float(row["power_mW"]))
            except Exception:
                continue
    if not ts:
        return [], []
    # sort
    xs = sorted(zip(ts, p_mw), key=lambda x: x[0])
    return [t for t, _ in xs], [p for _, p in xs]


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


def plot_power_trace(
    power_csv: str,
    run_log: str,
    segments: list[SegmentEnergy],
    outpath: str,
    baseline_s: float | None,
    clamp_delta: bool,
    align: bool,
    trim_baseline: bool,
):
    ts, p_mw = read_power_csv(power_csv)
    if not ts:
        raise SystemExit("No power samples found.")
    intervals = read_llm_intervals(run_log)
    if not intervals:
        raise SystemExit("No decode intervals found in run log.")

    shift_s = 0.0
    if align:
        first_start = min(s for s, _e, _m in intervals)
        i0 = min(range(len(ts)), key=lambda i: abs(ts[i] - first_start))
        shift_s = first_start - ts[i0]
        if abs(shift_s) > 2.0:
            shift_s = 0.0
        ts = [t + shift_s for t in ts]

    # Prefer baseline from segments CSV if present (keeps plots consistent with delta_energy_J).
    baseline_override_mw = None
    bs = [s.baseline_mw for s in segments if s.baseline_mw is not None and math.isfinite(s.baseline_mw)]
    if bs:
        baseline_override_mw = sum(bs) / len(bs)

    first_start = min(s for s, _e, _m in intervals)
    if baseline_override_mw is not None:
        baseline_mean = float(baseline_override_mw)
        baseline_mode = "from_llm_segments_csv"
    else:
        if baseline_s is None:
            baseline_end = first_start
            baseline_mode = "auto_pre_first_decode"
        else:
            baseline_end = ts[0] + float(baseline_s)
            baseline_mode = "fixed_seconds"
        vals = [mw for t, mw in zip(ts, p_mw) if t < baseline_end]
        baseline_mean = (sum(vals) / len(vals)) if vals else 0.0

    # Trim the baseline part of the trace (visualization only).
    trim_label = "none"
    trim_start = ts[0]
    if trim_baseline:
        workload_start = read_first_event_time(run_log, "workload_start")
        if workload_start is not None:
            trim_label = "workload_start"
            trim_start = workload_start
        elif baseline_s is not None:
            trim_label = "baseline_s"
            trim_start = ts[0] + float(baseline_s)
        else:
            trim_label = "first_decode"
            trim_start = first_start

    ts_plot = ts
    p_plot = p_mw
    if trim_baseline:
        keep = [(t, mw) for t, mw in zip(ts, p_mw) if t >= trim_start]
        if keep:
            ts_plot = [t for t, _mw in keep]
            p_plot = [mw for _t, mw in keep]

    t0 = ts_plot[0]
    x = [t - t0 for t in ts_plot]

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(14, 7), sharex=True)
    p_w = [mw / 1000.0 for mw in p_plot]
    dp_w = [(mw - baseline_mean) / 1000.0 for mw in p_plot]
    if clamp_delta:
        dp_w = [max(0.0, v) for v in dp_w]

    ax0.plot(x, p_w, linewidth=0.8, color="#333333", alpha=0.85, label="GPU power (W)")
    ax0.axhline(baseline_mean / 1000.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.9, label="baseline mean (W)")
    ax1.plot(x, dp_w, linewidth=0.9, color="#1f77b4", alpha=0.9, label="GPU delta power (W)")
    ax1.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)

    for (s, e, _m) in intervals:
        if e <= t0:
            continue
        xs = max(0.0, s - t0)
        xe = max(0.0, e - t0)
        ax0.axvspan(xs, xe, color="#1f77b4", alpha=0.12)
        ax1.axvspan(xs, xe, color="#1f77b4", alpha=0.12)

    trim_desc = f"{trim_label}@{trim_start - ts[0]:.1f}s" if trim_baseline else "none"
    ax0.set_title(
        f"GPU power with LLM decode windows (n={len(intervals)}), baseline={baseline_mode}, "
        f"align_shift_s={shift_s:.3f}, trim_baseline={trim_desc}"
    )
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


def plot_delta_power_only_trace(
    power_csv: str,
    run_log: str,
    segments: list[SegmentEnergy],
    outpath: str,
    baseline_s: float | None,
    clamp_delta: bool,
    align: bool,
    trim_baseline: bool,
):
    ts, p_mw = read_power_csv(power_csv)
    if not ts:
        raise SystemExit("No power samples found.")
    intervals = read_llm_intervals(run_log)
    if not intervals:
        raise SystemExit("No decode intervals found in run log.")

    shift_s = 0.0
    if align:
        first_start = min(s for s, _e, _m in intervals)
        i0 = min(range(len(ts)), key=lambda i: abs(ts[i] - first_start))
        shift_s = first_start - ts[i0]
        if abs(shift_s) > 2.0:
            shift_s = 0.0
        ts = [t + shift_s for t in ts]

    # Baseline from segments CSV if present.
    bs = [s.baseline_mw for s in segments if s.baseline_mw is not None and math.isfinite(s.baseline_mw)]
    baseline_mean = float(sum(bs) / len(bs)) if bs else 0.0
    baseline_w = baseline_mean / 1000.0

    first_start = min(s for s, _e, _m in intervals)
    trim_label = "none"
    trim_start = ts[0]
    if trim_baseline:
        workload_start = read_first_event_time(run_log, "workload_start")
        if workload_start is not None:
            trim_label = "workload_start"
            trim_start = workload_start
        elif baseline_s is not None:
            trim_label = "baseline_s"
            trim_start = ts[0] + float(baseline_s)
        else:
            trim_label = "first_decode"
            trim_start = first_start

    ts_plot = ts
    p_plot = p_mw
    if trim_baseline:
        keep = [(t, mw) for t, mw in zip(ts, p_mw) if t >= trim_start]
        if keep:
            ts_plot = [t for t, _mw in keep]
            p_plot = [mw for _t, mw in keep]

    t0 = ts_plot[0]
    x = [t - t0 for t in ts_plot]
    dp_w = [(mw - baseline_mean) / 1000.0 for mw in p_plot]
    if clamp_delta:
        dp_w = [max(0.0, v) for v in dp_w]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 4.3), sharex=True)
    ax.plot(x, dp_w, linewidth=0.9, color="#1f77b4", alpha=0.95, label="GPU")
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)

    for (s, e, _m) in intervals:
        if e <= t0:
            continue
        xs = max(0.0, s - t0)
        xe = max(0.0, e - t0)
        ax.axvspan(xs, xe, color="#1f77b4", alpha=0.10)

    trim_desc = f"{trim_label}@{trim_start - ts[0]:.1f}s" if trim_baseline else "none"
    ax.set_title(f"GPU power plot. Baseline substracted: {baseline_w:.3f} W (trim_baseline={trim_desc}, align_shift_s={shift_s:.3f})")
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, ncol=1)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    fig.savefig(os.path.splitext(outpath)[0] + ".pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot NVIDIA GPU power + decode-window segments.")
    ap.add_argument("--logdir", default="", help="Directory containing gpu_power_*.csv/workload_*.log/llm_segments_*.csv")
    ap.add_argument("--segments-csv", default="")
    ap.add_argument("--power-csv", default="")
    ap.add_argument("--run-log", default="")
    ap.add_argument("--baseline-s", default="auto", help="Baseline seconds for trace, or 'auto' (default: auto).")
    ap.add_argument("--no-align", action="store_true", help="Disable small timestamp alignment.")
    ap.add_argument("--clamp-delta", action="store_true", help="Clamp delta power/energy at 0 (visualization only).")
    ap.add_argument("--trim-baseline", action="store_true", help="Start trace at workload_start (or baseline end) instead of t=0.")
    ap.add_argument("--rails", default="", help="Comma-separated rails to plot (default: infer from segments).")
    ap.add_argument("--outdir", default="plots_gpu", help="Output directory.")
    args = ap.parse_args()

    seg = args.segments_csv.strip()
    pwr = args.power_csv.strip()
    run = args.run_log.strip()
    if args.logdir.strip():
        logdir = args.logdir.strip()

        def _pick_latest(prefix: str, suffix: str):
            cands = [os.path.join(logdir, f) for f in os.listdir(logdir) if f.startswith(prefix) and f.endswith(suffix)]
            return max(cands, key=lambda p: os.path.getmtime(p)) if cands else ""

        if not seg:
            seg = _pick_latest("llm_segments_", ".csv")
        if not pwr:
            pwr = _pick_latest("gpu_power_", ".csv")
        if not run:
            run = _pick_latest("workload_", ".log")

    if not seg or not os.path.exists(seg):
        raise SystemExit("Missing --segments-csv (or none found in --logdir).")
    if not pwr or not os.path.exists(pwr):
        raise SystemExit("Missing --power-csv (or none found in --logdir).")
    if not run or not os.path.exists(run):
        raise SystemExit("Missing --run-log (or none found in --logdir).")

    os.makedirs(args.outdir, exist_ok=True)
    segments = read_segments_csv(seg)
    rails = [r.strip() for r in args.rails.split(",") if r.strip()]
    if not rails:
        rails = sorted({s.rail for s in segments})

    baseline_s: float | None
    if isinstance(args.baseline_s, str) and args.baseline_s.strip().lower() in ("auto", ""):
        baseline_s = None
    else:
        baseline_s = float(args.baseline_s)

    print("Using logs:")
    print(f"  segments:  {seg}")
    print(f"  power_csv:  {pwr}")
    print(f"  run_log:    {run}")

    energy_plot = os.path.join(args.outdir, "energy_per_test.png")
    plot_energy_per_test(segments, energy_plot, rails=rails, clamp_delta=args.clamp_delta)
    power_plot = os.path.join(args.outdir, "power_per_test.png")
    plot_power_per_test(segments, power_plot, rails=rails, clamp_delta=args.clamp_delta)
    trace_plot = os.path.join(args.outdir, "power_trace_with_decode_windows.png")
    plot_power_trace(
        pwr,
        run,
        segments,
        trace_plot,
        baseline_s=baseline_s,
        clamp_delta=args.clamp_delta,
        align=(not args.no_align),
        trim_baseline=args.trim_baseline,
    )
    delta_only_plot = os.path.join(args.outdir, "delta_power_trace_only.png")
    plot_delta_power_only_trace(
        pwr,
        run,
        segments,
        delta_only_plot,
        baseline_s=baseline_s,
        clamp_delta=args.clamp_delta,
        align=(not args.no_align),
        trim_baseline=args.trim_baseline,
    )
    print("wrote:")
    print(f"  {energy_plot}")
    print(f"  {power_plot}")
    print(f"  {trace_plot}")
    print(f"  {delta_only_plot}")


if __name__ == "__main__":
    raise SystemExit(main())
