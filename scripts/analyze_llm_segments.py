#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
from dataclasses import dataclass
from typing import Any, Iterable


def _to_int(v: str | None) -> int | None:
    if v is None:
        return None
    v = str(v).strip()
    if not v or v.lower() == "none":
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _to_float(v: str | None) -> float | None:
    if v is None:
        return None
    v = str(v).strip()
    if not v or v.lower() == "none":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _summary(xs: list[float]) -> dict[str, float]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return {
            "n": 0,
            "sum": float("nan"),
            "mean": float("nan"),
            "stdev": float("nan"),
            "min": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    sxs = sorted(xs)
    return {
        "n": len(xs),
        "sum": float(sum(xs)),
        "mean": float(st.fmean(xs)),
        "stdev": float(st.pstdev(xs)) if len(xs) > 1 else 0.0,
        "min": float(sxs[0]),
        "p50": float(_percentile(sxs, 50)),
        "p90": float(_percentile(sxs, 90)),
        "p95": float(_percentile(sxs, 95)),
        "p99": float(_percentile(sxs, 99)),
        "max": float(sxs[-1]),
    }


@dataclass(frozen=True)
class SegmentRow:
    idx: int
    rail: str
    dur_s: float | None
    energy_j: float | None
    delta_energy_j: float | None
    baseline_mw: float | None
    prompt_chars: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    max_tokens: int | None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_power_mw(self) -> float | None:
        if self.energy_j is None or self.dur_s is None or self.dur_s <= 0:
            return None
        return (self.energy_j * 1000.0) / self.dur_s

    @property
    def avg_delta_power_mw(self) -> float | None:
        if self.delta_energy_j is None or self.dur_s is None or self.dur_s <= 0:
            return None
        return (self.delta_energy_j * 1000.0) / self.dur_s


def read_rows(path: str) -> list[SegmentRow]:
    rows: list[SegmentRow] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for rec in r:
            rows.append(
                SegmentRow(
                    idx=_to_int(rec.get("idx")) or 0,
                    rail=(rec.get("rail") or "").strip(),
                    dur_s=_to_float(rec.get("dur_s")),
                    energy_j=_to_float(rec.get("energy_J")),
                    delta_energy_j=_to_float(rec.get("delta_energy_J")),
                    baseline_mw=_to_float(rec.get("baseline_mW")),
                    prompt_chars=_to_int(rec.get("prompt_chars")),
                    prompt_tokens=_to_int(rec.get("prompt_tokens") or rec.get("end_prompt_tokens")),
                    completion_tokens=_to_int(rec.get("completion_tokens") or rec.get("end_completion_tokens")),
                    max_tokens=_to_int(rec.get("max_tokens")),
                )
            )
    return rows


def filter_rows(
    rows: Iterable[SegmentRow],
    rails: set[str] | None,
    skip_first: int,
    min_completion_tokens: int,
    drop_truncated: bool,
) -> list[SegmentRow]:
    out = []
    for row in rows:
        if rails is not None and row.rail not in rails:
            continue
        if row.idx < skip_first:
            continue
        ct = row.completion_tokens or 0
        if ct < min_completion_tokens:
            continue
        if drop_truncated and row.max_tokens is not None and row.completion_tokens is not None:
            if row.completion_tokens >= row.max_tokens:
                continue
        out.append(row)
    return out


def per_rail_stats(rows: list[SegmentRow]) -> dict[str, Any]:
    by_rail: dict[str, list[SegmentRow]] = {}
    for row in rows:
        by_rail.setdefault(row.rail, []).append(row)

    out: dict[str, Any] = {}
    for rail, rs in sorted(by_rail.items()):
        dur = [r.dur_s for r in rs if r.dur_s is not None]
        energy = [r.energy_j for r in rs if r.energy_j is not None]
        delta = [r.delta_energy_j for r in rs if r.delta_energy_j is not None]
        p_mw = [r.avg_power_mw for r in rs if r.avg_power_mw is not None]
        dp_mw = [r.avg_delta_power_mw for r in rs if r.avg_delta_power_mw is not None]

        # Energy per token (only meaningful for VIN rail usually, but compute anyway)
        e_per_out_tok = []
        de_per_out_tok = []
        e_per_total_tok = []
        de_per_total_tok = []
        for r in rs:
            if r.energy_j is None or r.delta_energy_j is None:
                continue
            if r.completion_tokens and r.completion_tokens > 0:
                e_per_out_tok.append(r.energy_j / r.completion_tokens)
                de_per_out_tok.append(r.delta_energy_j / r.completion_tokens)
            tt = r.total_tokens
            if tt and tt > 0:
                e_per_total_tok.append(r.energy_j / tt)
                de_per_total_tok.append(r.delta_energy_j / tt)

        out[rail] = {
            "count_rows": len(rs),
            "duration_s": _summary(dur),
            "energy_J": _summary(energy),
            "delta_energy_J": _summary(delta),
            "avg_power_mW": _summary(p_mw),
            "avg_delta_power_mW": _summary(dp_mw),
            "energy_J_per_output_token": _summary(e_per_out_tok),
            "delta_energy_J_per_output_token": _summary(de_per_out_tok),
            "energy_J_per_total_token": _summary(e_per_total_tok),
            "delta_energy_J_per_total_token": _summary(de_per_total_tok),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyze llm_segments_*.csv produced by profile_jetson_power.sh --segment-llm.")
    ap.add_argument("csv", help="Path to llm_segments_<ts>.csv")
    ap.add_argument("--rails", default="", help="Comma-separated rails to include (default: all in file).")
    ap.add_argument("--skip-first", type=int, default=1, help="Skip first N idx values (warmup). Default: 1")
    ap.add_argument("--min-completion-tokens", type=int, default=1, help="Filter out rows with fewer output tokens. Default: 1")
    ap.add_argument("--drop-truncated", action="store_true", help="Drop rows where completion_tokens >= max_tokens (often truncated).")
    ap.add_argument("--out-json", default="", help="Write full stats JSON to this path.")
    ap.add_argument("--out-md", default="", help="Write a compact Markdown summary to this path.")
    args = ap.parse_args()

    path = args.csv
    rows = read_rows(path)
    if not rows:
        raise SystemExit(f"No rows read from {path}")

    rails = None
    if args.rails.strip():
        rails = {r.strip() for r in args.rails.split(",") if r.strip()}

    rows_f = filter_rows(
        rows,
        rails=rails,
        skip_first=max(0, args.skip_first),
        min_completion_tokens=max(0, args.min_completion_tokens),
        drop_truncated=bool(args.drop_truncated),
    )
    if not rows_f:
        raise SystemExit("No rows left after filtering. Try reducing --skip-first/--min-completion-tokens.")

    stats = {
        "input_csv": os.path.abspath(path),
        "filters": {
            "rails": sorted(list(rails)) if rails is not None else None,
            "skip_first": args.skip_first,
            "min_completion_tokens": args.min_completion_tokens,
            "drop_truncated": bool(args.drop_truncated),
        },
        "rows_total": len(rows),
        "rows_used": len(rows_f),
        "per_rail": per_rail_stats(rows_f),
    }

    # Console output (compact)
    print(f"file: {path}")
    print(f"rows_used: {stats['rows_used']} (of {stats['rows_total']})  skip_first={args.skip_first}  min_completion_tokens={args.min_completion_tokens}  drop_truncated={args.drop_truncated}")
    for rail, rst in stats["per_rail"].items():
        e = rst["delta_energy_J"]
        p = rst["avg_delta_power_mW"]
        ept = rst["delta_energy_J_per_output_token"]
        print(
            f"{rail}: delta_J mean={e['mean']:.4f} p50={e['p50']:.4f} p95={e['p95']:.4f} "
            f"| delta_mW mean={p['mean']:.1f} "
            f"| delta_J/out_tok mean={ept['mean']:.6f}"
        )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        print(f"wrote: {args.out_json}")

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(f"# LLM Segment Stats\n\n")
            f.write(f"- file: `{stats['input_csv']}`\n")
            f.write(f"- rows_used: `{stats['rows_used']}` / `{stats['rows_total']}`\n")
            f.write(f"- skip_first: `{args.skip_first}`\n")
            f.write(f"- min_completion_tokens: `{args.min_completion_tokens}`\n")
            f.write(f"- drop_truncated: `{args.drop_truncated}`\n\n")
            for rail, rst in stats["per_rail"].items():
                f.write(f"## {rail}\n\n")
                for k in (
                    "duration_s",
                    "energy_J",
                    "delta_energy_J",
                    "avg_power_mW",
                    "avg_delta_power_mW",
                    "delta_energy_J_per_output_token",
                    "delta_energy_J_per_total_token",
                ):
                    s = rst[k]
                    f.write(f"- `{k}`: n={s['n']} mean={s['mean']:.6g} p50={s['p50']:.6g} p95={s['p95']:.6g} max={s['max']:.6g}\n")
                f.write("\n")
        print(f"wrote: {args.out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
