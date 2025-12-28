#!/usr/bin/env python3
import datetime as dt
import re
import time
from collections.abc import Iterable, Iterator
from typing import NamedTuple


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


class TegraSample(NamedTuple):
    t_epoch_s: float
    dt_s: float
    rails_mw: dict[str, int]


def iter_tegrastats_samples(path: str, rails: list[str]) -> Iterator[TegraSample]:
    """
    Iterate tegrastats samples with per-sample dt derived from the timestamp buckets.

    tegrastats timestamps are only 1-second resolution, but it can print many samples
    within the same second when using sub-second intervals. To reconstruct time, we:
      - group consecutive lines by their wall-clock second
      - distribute samples evenly within the time span to the next second bucket

    This avoids assuming that the requested `--interval` was actually achieved (it often
    isn't under load), which otherwise corrupts baseline selection and energy estimates.
    """
    rails = [r.strip() for r in rails if r.strip()]
    if not rails:
        return

    rail_res = {rail: re.compile(rf"{re.escape(rail)}\s+(\d+)mW\b") for rail in rails}
    last_vals: dict[str, int] = {rail: 0 for rail in rails}

    bucket_epoch: float | None = None
    bucket_rows: list[dict[str, int]] = []
    last_span_s = 1.0

    def flush_bucket(next_epoch: float | None):
        nonlocal bucket_epoch, bucket_rows, last_span_s
        if bucket_epoch is None or not bucket_rows:
            return
        if next_epoch is None:
            span_s = last_span_s
        else:
            span_s = float(next_epoch - bucket_epoch)
            if span_s <= 0:
                span_s = last_span_s
        if span_s <= 0:
            span_s = 1.0
        last_span_s = span_s
        dt_s = span_s / len(bucket_rows)
        for i, rails_mw in enumerate(bucket_rows):
            yield TegraSample(t_epoch_s=float(bucket_epoch) + (i * dt_s), dt_s=dt_s, rails_mw=rails_mw)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            epoch = parse_tegrastats_time_to_epoch(line)
            if epoch is None:
                continue

            if bucket_epoch is None:
                bucket_epoch = float(epoch)
            elif float(epoch) != bucket_epoch:
                yield from flush_bucket(float(epoch))
                bucket_epoch = float(epoch)
                bucket_rows = []

            for rail, rx in rail_res.items():
                m = rx.search(line)
                if m:
                    last_vals[rail] = int(m.group(1))
            bucket_rows.append(dict(last_vals))

    yield from flush_bucket(None)


def weighted_mean_in_window(samples: Iterable[TegraSample], rail: str, t0: float, t1: float) -> float:
    """
    Time-weighted mean power (mW) for a rail over [t0, t1).
    """
    if t1 <= t0:
        return 0.0
    num = 0.0
    den = 0.0
    for s in samples:
        a = s.t_epoch_s
        b = s.t_epoch_s + s.dt_s
        if b <= t0:
            continue
        if a >= t1:
            break
        overlap = max(0.0, min(b, t1) - max(a, t0))
        if overlap <= 0:
            continue
        num += float(s.rails_mw.get(rail, 0)) * overlap
        den += overlap
    return (num / den) if den > 0 else 0.0

