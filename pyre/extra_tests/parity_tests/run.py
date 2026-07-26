#!/usr/bin/env python3
"""Parity baseline runner.

Runs every script under `pyre/extra_tests/parity_tests/` against:
  - CPython (the system `python3`),
  - the pyre-dynasm binary (release build),
  - the pyre-cranelift binary (release build, if present).

A script passes when:
  - the process exits with code 0,
  - the last non-empty stdout line equals "OK".

Any divergence between CPython and a pyre backend is a parity
regression: the runtime has drifted from CPython observable
semantics.

Usage:
    python3 pyre/extra_tests/parity_tests/run.py
        [--dynasm-only|--cranelift-only] [--gc-poison]

`--gc-poison` fills reclaimed nursery bytes with a poison pattern for the
pyre backends. A dangling reference into reclaimed nursery memory usually
still decodes as a plausible object, so without poison that whole class of
GC defect passes silently; with it the run aborts at the first stale read.

Exit code is 0 iff every (script, backend) pair passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
TARGET_RELEASE = ROOT / "target" / "release"

EXE = ".exe" if sys.platform == "win32" else ""


def _scripts() -> list[Path]:
    out = []
    for p in sorted(HERE.glob("*.py")):
        if p.name == "run.py":
            continue
        out.append(p)
    return out


def _run(cmd: list[str], script: Path, env: dict[str, str] | None) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd + [str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=None if env is None else {**os.environ, **env},
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = proc.stdout
    err = proc.stderr
    lines = [line for line in out.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    ok = proc.returncode == 0 and last == "OK"
    detail = "" if ok else f"rc={proc.returncode} last={last!r} stderr={err.strip()!r}"
    return ok, detail


def _runners(
    only_dynasm: bool, only_cranelift: bool, gc_poison: bool
) -> list[tuple[str, list[str], dict[str, str] | None]]:
    runners: list[tuple[str, list[str], dict[str, str] | None]] = []
    cpython = os.environ.get("PYRE_CHECK_PYTHON3") or "python3"
    runners.append(("cpython", [cpython], None))
    pyre_env = {"MAJIT_GC_NURSERY_POISON": "1"} if gc_poison else None
    dynasm = TARGET_RELEASE / f"pyre-dynasm{EXE}"
    cranelift = TARGET_RELEASE / f"pyre-cranelift{EXE}"
    if not only_cranelift and dynasm.exists():
        runners.append(("dynasm", [str(dynasm)], pyre_env))
    if not only_dynasm and cranelift.exists():
        runners.append(("cranelift", [str(cranelift)], pyre_env))
    return runners


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynasm-only", action="store_true")
    parser.add_argument("--cranelift-only", action="store_true")
    parser.add_argument("--gc-poison", action="store_true")
    args = parser.parse_args()

    runners = _runners(args.dynasm_only, args.cranelift_only, args.gc_poison)
    scripts = _scripts()
    if not scripts:
        print("no parity test scripts found", file=sys.stderr)
        return 1

    print(f"runners: {[name for name, _, _ in runners]}")
    print(f"scripts: {len(scripts)}")
    print()

    fail = 0
    for script in scripts:
        name = script.name
        row: list[str] = [f"  {name:<36s}"]
        for backend, cmd, env in runners:
            ok, detail = _run(cmd, script, env)
            mark = "OK" if ok else "FAIL"
            row.append(f"{backend}={mark}")
            if not ok:
                fail += 1
                print(f"    {backend} {name}: {detail}", file=sys.stderr)
        print(" ".join(row))

    print()
    if fail:
        print(f"{fail} failure(s)", file=sys.stderr)
        return 1
    print("all parity tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
