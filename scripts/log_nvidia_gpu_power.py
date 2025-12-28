#!/usr/bin/env python3
import argparse
import csv
import signal
import subprocess
import sys
import time


STOP = False


def _on_sig(_sig, _frame):
    global STOP
    STOP = True


def try_nvml_power(gpu_index: int) -> callable | None:
    """
    Optional fast path via pynvml (if installed). Returns a callable that returns power_mW.
    """
    try:
        import pynvml  # type: ignore
    except Exception:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))

        def _read() -> int:
            return int(pynvml.nvmlDeviceGetPowerUsage(handle))

        return _read
    except Exception:
        return None


def read_power_mw_via_nvidia_smi(gpu_index: int) -> int:
    cmd = [
        "nvidia-smi",
        "-i",
        str(int(gpu_index)),
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    # e.g. "123.45" (W)
    w = float(out.splitlines()[-1].strip())
    return int(round(w * 1000.0))


def main() -> int:
    ap = argparse.ArgumentParser(description="Log NVIDIA GPU power to CSV (timestamped on the host).")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    ap.add_argument("--interval-ms", type=int, default=100, help="Sampling interval in ms (default: 100)")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    args = ap.parse_args()

    if args.interval_ms <= 0:
        raise SystemExit("--interval-ms must be > 0")

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    reader = try_nvml_power(args.gpu)
    mode = "pynvml" if reader is not None else "nvidia-smi"
    if reader is None:
        reader = lambda: read_power_mw_via_nvidia_smi(args.gpu)

    dt_s = args.interval_ms / 1000.0
    t_next = time.monotonic()

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_epoch_s", "power_mW", "gpu_index", "source"])

        while not STOP:
            # Keep a steady cadence based on monotonic time.
            now_m = time.monotonic()
            if now_m < t_next:
                time.sleep(min(0.05, t_next - now_m))
                continue
            t_next = max(t_next + dt_s, now_m)

            try:
                p_mw = int(reader())
            except Exception:
                # If sampling fails transiently, skip this tick.
                continue

            w.writerow([f"{time.time():.6f}", p_mw, int(args.gpu), mode])
            f.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

