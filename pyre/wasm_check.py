#!/usr/bin/env python3
"""Compare pyre's wasm backend across host engines: wasmtime vs wasmi vs wasmi+majit.

Usage:
    python pyre/wasm_check.py [--reps N] [--timeout T] [BENCH_GLOB ...]

Examples:
    python pyre/wasm_check.py                  # all pyre/bench/*.py
    python pyre/wasm_check.py int_loop fib     # scripts matching these substrings
    python pyre/wasm_check.py --reps 3         # best-of-3 per (bench, engine)

The wasm module is built once; only the native host interpreting it changes.
Outputs are required to match across engines — a mismatch means the majit tier
miscompiled and is flagged.

Requires pyre-wasm-runner to be built:
    cargo build --release -p pyre-wasm-runner
"""

from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
from pathlib import Path

BENCH_DIR = "pyre/bench"
EXE = ".exe" if sys.platform == "win32" else ""
# Per-bench timeout (seconds). Override with --timeout.
DEFAULT_TIMEOUT = 15

# Host-engine configurations:
# (label, PYRE_WASM_ENGINE value, WASMI_NO_MAJIT value or None)
#
# Note: wasmi+majit requires building pyre-wasm-runner against the patched wasmi
# (Cargo.toml path override pointing to ./wasmi/).  With stock wasmi 1.1,
# WASMI_NO_MAJIT is a no-op — both "wasmi-stock" and "wasmi+majit" rows will
# report identical (stock interpreter) performance.
ENGINES = [
    ("wasmtime",    "wasmtime", None),
    ("wasmi-stock", "wasmi",    "1"),
    ("wasmi+majit", "wasmi",    None),
]


# ── Formatting helpers ───────────────────────────────────────────────

def red(s):    return f"\033[31m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"


# ── Child-process user CPU time ──────────────────────────────────────

def run_timed(args, timeout_s=None, env=None):
    """Run *args*, return (stdout_str, user_cpu_seconds, returncode, stderr_str)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        proc = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return "", 0.0, 124, ""
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    utime = max(after.ru_utime - before.ru_utime, 0.0)
    return (
        proc.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n"),
        utime,
        proc.returncode,
        proc.stderr.decode("utf-8", errors="replace"),
    )


# ── Main ─────────────────────────────────────────────────────────────

def find_runner() -> str:
    for candidate in [
        f"./target/release/pyre-wasm-runner{EXE}",
        f"target/release/pyre-wasm-runner{EXE}",
    ]:
        if os.access(candidate, os.X_OK):
            return candidate
    print("ERROR: pyre-wasm-runner not found. Build with:")
    print("  cargo build --release -p pyre-wasm-runner")
    sys.exit(1)


def collect_benches(filters: list[str]) -> list[Path]:
    """Glob top-level pyre/bench/*.py, optionally filtering by substring.

    Only top-level scripts are collected (not pyre/bench/synth/ or subdirs).
    Synthetic benchmarks under synth/ are pyre-interpreter-specific and are
    not expected to run correctly under the wasm path.
    """
    all_scripts = sorted(Path(BENCH_DIR).glob("*.py"))
    # Exclude tmp_ files
    all_scripts = [s for s in all_scripts if not s.name.startswith("tmp_")]
    if not filters:
        return all_scripts
    matched = []
    for s in all_scripts:
        if any(f in s.stem for f in filters):
            matched.append(s)
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filters", nargs="*",
                        help="bench name substrings to select (default: all)")
    parser.add_argument("--reps", type=int, default=1,
                        help="repetitions per (bench, engine); report best (default: 1)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"per-run timeout in seconds (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()

    runner = find_runner()
    benches = collect_benches(args.filters)
    if not benches:
        print("No benchmarks found.")
        sys.exit(1)

    reps = max(1, args.reps)
    timeout = args.timeout
    labels = [c[0] for c in ENGINES]
    base_env = dict(os.environ)

    print()
    print(bold("wasm engine comparison — pyre module under each host"))
    print(dim(f"reps={reps} (best user-CPU seconds); timeout={timeout}s"))
    print()
    header = f"{'bench':<20s}" + "".join(f"{l:>14s}" for l in labels) + f"{'majit spd':>11s}  correct"
    print(header)
    print("-" * len(header))

    any_wrong = False
    for script in benches:
        name = script.stem
        times: dict[str, tuple[float | None, int]] = {}
        outputs: dict[str, str | None] = {}

        for label, engine, no_majit in ENGINES:
            env = dict(base_env)
            env["PYRE_WASM_ENGINE"] = engine
            if no_majit:
                env["WASMI_NO_MAJIT"] = no_majit
            else:
                env.pop("WASMI_NO_MAJIT", None)

            best, out, code = None, None, 0
            for _ in range(reps):
                o, t, c, _e = run_timed([runner, str(script)], timeout_s=timeout, env=env)
                if c != 0:
                    code = c
                    break
                out = o
                best = t if best is None else min(best, t)
            times[label] = (best, code)
            outputs[label] = out

        # Correctness: all engines that succeeded must produce identical output.
        # A crash or timeout also counts as a mismatch — the bench is flagged.
        ref, ok = None, True
        any_crash = any(times[l][1] != 0 for l in labels)
        if any_crash:
            ok = False
        for label in labels:
            if times[label][1] == 0 and outputs[label] is not None:
                if ref is None:
                    ref = outputs[label]
                elif outputs[label] != ref:
                    ok = False

        cells = []
        for label in labels:
            best, code = times[label]
            if code == 124:
                cells.append("TIMEOUT")
            elif code != 0:
                cells.append(f"CRASH({code})")
            else:
                cells.append(f"{best:.3f}s")

        stock_t, s_code = times["wasmi-stock"]
        majit_t, m_code = times["wasmi+majit"]
        if s_code == 0 and m_code == 0 and stock_t and majit_t:
            spd = f"{stock_t / majit_t:.2f}x"
        else:
            spd = "-"
        correct = green("yes") if ok else red("NO")
        any_wrong = any_wrong or not ok
        print(f"{name:<20s}" + "".join(f"{c:>14s}" for c in cells) + f"{spd:>11s}  {correct}")

    print()
    print(dim("majit spd = wasmi-stock / wasmi+majit user-CPU (higher = tier helps more)"))
    sys.exit(1 if any_wrong else 0)


if __name__ == "__main__":
    main()
