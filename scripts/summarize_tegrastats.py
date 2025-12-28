#!/usr/bin/env python3
import argparse
import sys
try:
    from tegrastats_utils import iter_tegrastats_samples
except ModuleNotFoundError:  # pragma: no cover
    from scripts.tegrastats_utils import iter_tegrastats_samples


def summarize(path: str, rail: str, baseline_s: float | None, baseline_samples: int, baseline_estimator: str):
    samples = iter_tegrastats_samples(path, rails=[rail])

    t_start: float | None = None
    n_total = 0
    total_energy_mw_s = 0.0
    total_time_s = 0.0
    total_peak_mw = 0

    # Baseline/run stats
    b_n = 0
    b_energy_mw_s = 0.0
    b_time_s = 0.0
    b_peak_mw = 0

    r_n = 0
    r_energy_mw_s = 0.0
    r_time_s = 0.0
    r_peak_mw = 0

    baseline_end: float | None = None
    baseline_vals: list[int] = []

    for s in samples:
        p_mw = int(s.rails_mw.get(rail, 0))
        if t_start is None:
            t_start = float(s.t_epoch_s)
            if baseline_s is not None and baseline_s > 0:
                baseline_end = t_start + float(baseline_s)

        n_total += 1
        total_energy_mw_s += p_mw * s.dt_s
        total_time_s += s.dt_s
        total_peak_mw = max(total_peak_mw, p_mw)

        in_baseline = False
        if baseline_end is not None:
            in_baseline = (s.t_epoch_s + s.dt_s) <= baseline_end
        elif baseline_samples > 0:
            in_baseline = n_total <= baseline_samples

        if in_baseline:
            baseline_vals.append(p_mw)
            b_n += 1
            b_energy_mw_s += p_mw * s.dt_s
            b_time_s += s.dt_s
            b_peak_mw = max(b_peak_mw, p_mw)
        else:
            r_n += 1
            r_energy_mw_s += p_mw * s.dt_s
            r_time_s += s.dt_s
            r_peak_mw = max(r_peak_mw, p_mw)

    if n_total == 0 or total_time_s <= 0:
        return None

    total = {
        "samples": n_total,
        "dur_s": total_time_s,
        "avg_mw": total_energy_mw_s / total_time_s,
        "peak_mw": total_peak_mw,
        "energy_j": total_energy_mw_s / 1000.0,
    }

    baseline_mean_mw = (b_energy_mw_s / b_time_s) if b_time_s > 0 else 0.0
    baseline_used_mw = baseline_mean_mw
    if baseline_vals and baseline_estimator != "mean":
        v = sorted(baseline_vals)
        if baseline_estimator == "min":
            baseline_used_mw = float(v[0])
        elif baseline_estimator == "p10":
            baseline_used_mw = float(v[int((len(v) - 1) * 0.10)])
        elif baseline_estimator == "p50":
            baseline_used_mw = float(v[int((len(v) - 1) * 0.50)])

    run = {
        "samples": r_n,
        "dur_s": r_time_s,
        "avg_mw": (r_energy_mw_s / r_time_s) if r_time_s > 0 else 0.0,
        "peak_mw": r_peak_mw,
        "energy_j": r_energy_mw_s / 1000.0,
        "baseline_mw": baseline_used_mw,
        "baseline_mean_mw": baseline_mean_mw,
        "delta_energy_j": (r_energy_mw_s - (baseline_used_mw * r_time_s)) / 1000.0,
    }

    return total, run


def main():
    ap = argparse.ArgumentParser(description="Summarize tegrastats rail power (avg/peak/energy; optional baseline subtraction).")
    ap.add_argument("--tegrastats-log", required=True)
    ap.add_argument("--interval-ms", type=int, required=True, help="Nominal tegrastats interval (kept for compatibility; analysis derives timing from timestamps).")
    ap.add_argument("--baseline-s", type=float, default=None, help="Baseline duration in seconds (preferred over --baseline-samples).")
    ap.add_argument("--baseline-samples", type=int, default=0)
    ap.add_argument(
        "--baseline-estimator",
        default="mean",
        choices=("mean", "p10", "p50", "min"),
        help="How to estimate baseline power from the baseline window (default: mean).",
    )
    ap.add_argument("--rails", default="VIN_SYS_5V0,VDD_GPU_SOC,VDD_CPU_CV")
    args = ap.parse_args()

    rails = [r.strip() for r in args.rails.split(",") if r.strip()]

    missing = []
    for rail in rails:
        out = summarize(args.tegrastats_log, rail, args.baseline_s, args.baseline_samples, args.baseline_estimator)
        if out is None:
            missing.append(rail)
            continue
        total, run = out
        print(
            f"{rail} total: samples={total['samples']} dur_s={total['dur_s']:.3f} "
            f"avg_mW={total['avg_mw']:.1f} peak_mW={total['peak_mw']} energy_J={total['energy_j']:.3f}"
        )
        if (args.baseline_samples > 0) or (args.baseline_s and args.baseline_s > 0):
            extra = ""
            if args.baseline_estimator != "mean":
                extra = f" (baseline_mean_mW={run['baseline_mean_mw']:.1f}, estimator={args.baseline_estimator})"
            print(
                f"{rail} run:   samples={run['samples']} dur_s={run['dur_s']:.3f} "
                f"avg_mW={run['avg_mw']:.1f} peak_mW={run['peak_mw']} energy_J={run['energy_j']:.3f} "
                f"baseline_mW={run['baseline_mw']:.1f} delta_energy_J={run['delta_energy_j']:.3f}{extra}"
            )

    if missing:
        print(f"Warning: no samples found for rails: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
