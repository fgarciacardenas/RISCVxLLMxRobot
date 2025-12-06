#!/usr/bin/env python3
"""
Summarize decision tester logs into a table for Benchmarks.xlsx.

It scans *_samples.json files under tests/decision_tester/logs/report_logs,
computes per-run metrics, and optionally emits grouped averages per
(Model, RAG, Device, Quantized, Binary). Defaults assume you run from the
repo root.

Example:
    python scripts/summarize_benchmarks.py \
        --logs src/LLMxRobot/tests/decision_tester/logs/report_logs \
        --csv-runs benchmarks_runs.csv \
        --csv-agg benchmarks_agg.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


def infer_rag_label(entry: Dict) -> str:
    rag_mode = (entry.get("rag_mode") or "").lower()
    if rag_mode == "online":
        return "GPT-4o"
    if rag_mode == "offline":
        return "BAAI"
    return "None"


def infer_device(model_name: str, run_dir: Path) -> str:
    blob = f"{model_name} {' '.join(run_dir.parts)}".lower()
    if "gguf" in blob:
        return "llama.cpp (GGUF)"
    if "local_" in blob or "axelera" in blob:
        return "Axelera"
    return "GPU"


def infer_quantized(model_name: str, run_dir: Path) -> str:
    blob = f"{model_name} {' '.join(run_dir.parts)}".lower()
    quant_markers = ("gguf", "q4", "q5", "int4", "int8", "quant")
    return "Yes" if any(marker in blob for marker in quant_markers) else "No"


def infer_binary(run_dir: Path) -> str:
    return "Yes" if any("binary" in part.lower() for part in run_dir.parts) else "No"


def load_entries(files: Iterable[Path]) -> List[Dict]:
    entries: List[Dict] = []
    for fp in files:
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def summarize_run(run_dir: Path) -> Optional[Dict]:
    json_files = sorted(run_dir.glob("*_samples.json"))
    entries = load_entries(json_files)
    if not entries:
        return None

    parsed = [
        e for e in entries
        if isinstance(e.get("sanitized_output"), bool) and isinstance(e.get("expected_output"), bool)
    ]

    # Accuracy where unparsed = incorrect
    total = len(entries)
    per_case_totals: Dict[str, int] = {}
    per_case_correct_all: Dict[str, int] = {}

    # Accuracy on parsed only
    correct_flags = []
    per_case_parsed: Dict[str, List[bool]] = {}

    for e in entries:
        case = e.get("test_case", "<unknown>")
        per_case_totals[case] = per_case_totals.get(case, 0) + 1

        parsed_ok = (
            isinstance(e.get("sanitized_output"), bool)
            and isinstance(e.get("expected_output"), bool)
        )
        if parsed_ok:
            is_correct = e["sanitized_output"] == e["expected_output"]
            correct_flags.append(is_correct)
            per_case_parsed.setdefault(case, []).append(is_correct)
            if is_correct:
                per_case_correct_all[case] = per_case_correct_all.get(case, 0) + 1
        else:
            # unparsed counts as incorrect for the "all" denominator
            per_case_correct_all[case] = per_case_correct_all.get(case, 0)

    parsed_count = len(correct_flags)
    acc_micro_parsed = (sum(correct_flags) / parsed_count) if parsed_count else 0.0
    acc_macro_parsed = (
        sum(sum(v) / len(v) for v in per_case_parsed.values()) / len(per_case_parsed)
        if per_case_parsed else 0.0
    )

    acc_micro_all = (sum(per_case_correct_all.values()) / total) if total else 0.0
    acc_macro_all = (
        sum(per_case_correct_all[c] / per_case_totals[c] for c in per_case_totals) / len(per_case_totals)
        if per_case_totals else 0.0
    )

    structure_rate = (len(parsed) / len(entries)) if entries else 0.0

    output_tokens_all = [int(e.get("output_tokens", 0)) for e in entries]
    output_tokens_parsed = [int(e.get("output_tokens", 0)) for e in parsed]

    first = entries[0]
    # Use the parent directory name (e.g., group folder under report_logs) as the model label
    model_name = run_dir.parent.name

    return {
        "Model": model_name,
        "RAG": infer_rag_label(first),
        "Device": infer_device(model_name, run_dir),
        "Quantized": infer_quantized(model_name, run_dir),
        "Binary": infer_binary(run_dir),
        "Accuracy micro parsed (%)": round(acc_micro_parsed * 100, 2),
        "Accuracy macro parsed (%)": round(acc_macro_parsed * 100, 2),
        "Accuracy micro all (%)": round(acc_micro_all * 100, 2),
        "Accuracy macro all (%)": round(acc_macro_all * 100, 2),
        "Structure followed (%)": round(structure_rate * 100, 2),
        "Output tokens": round(mean(output_tokens_all), 2) if output_tokens_all else 0.0,
        "Output tokens (filt)": round(mean(output_tokens_parsed), 2) if output_tokens_parsed else 0.0,
        "Run": str(run_dir),
    }


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = {}
    for row in rows:
        key = (row["Model"], row["RAG"], row["Device"], row["Quantized"], row["Binary"])
        grouped.setdefault(key, []).append(row)

    agg_rows: List[Dict] = []
    for key, items in sorted(grouped.items()):
        agg_rows.append({
            "Model": key[0],
            "RAG": key[1],
            "Device": key[2],
            "Quantized": key[3],
            "Binary": key[4],
            "Runs": len(items),
            "Accuracy micro parsed (%)": round(mean(i["Accuracy micro parsed (%)"] for i in items), 2),
            "Accuracy macro parsed (%)": round(mean(i["Accuracy macro parsed (%)"] for i in items), 2),
            "Accuracy micro all (%)": round(mean(i["Accuracy micro all (%)"] for i in items), 2),
            "Accuracy macro all (%)": round(mean(i["Accuracy macro all (%)"] for i in items), 2),
            "Structure followed (%)": round(mean(i["Structure followed (%)"] for i in items), 2),
            "Output tokens": round(mean(i["Output tokens"] for i in items), 2),
            "Output tokens (filt)": round(mean(i["Output tokens (filt)"] for i in items), 2),
        })
    return agg_rows


def print_table(rows: List[Dict], title: str) -> None:
    if not rows:
        print(f"{title}: no rows")
        return

    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    print(f"\n{title}")
    print("-" * sum(widths.values()))
    header_line = " | ".join(f"{h:{widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(f"{str(row[h]):{widths[h]}}" for h in headers))


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize decision tester logs for Benchmarks.xlsx.")
    parser.add_argument("--logs", type=Path,
                        default=Path("src/LLMxRobot/tests/decision_tester/logs/report_logs"),
                        help="Directory containing per-run logs.")
    parser.add_argument("--csv-runs", type=Path, default=None,
                        help="Optional path to write per-run metrics as CSV.")
    parser.add_argument("--csv-agg", type=Path, default=None,
                        help="Optional path to write grouped averages as CSV.")
    args = parser.parse_args()

    if not args.logs.exists():
        raise SystemExit(f"Log directory not found: {args.logs}")

    run_dirs = sorted({fp.parent for fp in args.logs.rglob("*_samples.json")})
    rows = [row for rd in run_dirs if (row := summarize_run(rd))]

    if not rows:
        raise SystemExit("No *_samples.json files found to summarize.")

    print_table(rows, "Per-run metrics")
    agg_rows = aggregate_rows(rows)
    print_table(agg_rows, "Grouped averages (by model/RAG/device/quant/binary)")

    if args.csv_runs:
        write_csv(rows, args.csv_runs)
        print(f"\nWrote per-run CSV to {args.csv_runs}")
    if args.csv_agg:
        write_csv(agg_rows, args.csv_agg)
        print(f"Wrote grouped CSV to {args.csv_agg}")


if __name__ == "__main__":
    main()
