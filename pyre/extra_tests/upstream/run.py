#!/usr/bin/env python3
"""Runner for the vendored PyPy `extra_tests/` tree at the repository root.

Those files are upstream's, not ours: they are left where they are and run in
place through `driver.py`, which supplies the small slice of the pytest API
they use.  Nothing is copied into `pyre/extra_tests/`.

Only the files named in `ENABLED` run.  The tree carries 60+ files covering
surface pyre has not reached yet, and a runner that reddens on all of them
would report nothing; a file joins the list once it passes under CPython and
under the pyre backends.

Usage:
    python3 pyre/extra_tests/upstream/run.py [--dynasm-only|--cranelift-only]
                                             [--cpython-only]
                                             [--filter SUBSTRING]
                                             [--timeout SECONDS]
                                             [--all] [--list] [-v]

Exit code is 0 iff every (file, backend) pair passed.  `--all` ignores
`ENABLED` and runs the whole vendored tree; it is a survey switch, not a gate.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
UPSTREAM = ROOT / "extra_tests"
DRIVER = HERE / "driver.py"
TARGET_RELEASE = ROOT / "target" / "release"

EXE = ".exe" if sys.platform == "win32" else ""

# Vendored upstream files that pass here, each with the `os.name` it needs
# (None runs everywhere).  Add a file after checking it passes under CPython
# and every pyre backend; `--all` shows what the rest do.
ENABLED: list[tuple[str, str | None]] = [
    # `spawnv`/`spawnve` are noop stubs under `nt` (interp_posix.rs:1292 — the
    # module owns those names on Windows and there is no implementation behind
    # them), so `test_spawnv`'s `ret == 42` is not a Windows expectation.
    ("test_os.py", "posix"),
]


def _files(args: argparse.Namespace) -> tuple[list[Path], int]:
    """Selected files, and how many `ENABLED` entries this platform skipped."""
    if args.all:
        candidates = [(p.name, None) for p in sorted(UPSTREAM.glob("test_*.py"))]
    else:
        candidates = ENABLED
    out: list[Path] = []
    off_platform = 0
    for name, os_name in candidates:
        if os_name is not None and os.name != os_name:
            off_platform += 1
            continue
        p = UPSTREAM / name
        if not p.exists():
            print(f"missing vendored test file: {p}", file=sys.stderr)
            continue
        if args.filter and args.filter not in p.name:
            continue
        out.append(p)
    return out, off_platform


def _run(cmd: list[str], path: Path, timeout: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd + [str(DRIVER), str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode == 0:
        tail = [line for line in proc.stdout.splitlines() if line.strip()]
        return True, tail[-1] if tail else ""
    err = proc.stderr.strip().splitlines()
    out = proc.stdout.strip().splitlines()
    detail = out[-1] if out else (err[-1] if err else "")
    return False, f"rc={proc.returncode} {detail}"


def _runners(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    runners: list[tuple[str, list[str]]] = []
    cpython = os.environ.get("PYRE_CHECK_PYTHON3") or "python3"
    if not args.dynasm_only and not args.cranelift_only:
        runners.append(("cpython", [cpython]))
    if args.cpython_only:
        return runners
    dynasm = TARGET_RELEASE / f"pyre-dynasm{EXE}"
    cranelift = TARGET_RELEASE / f"pyre-cranelift{EXE}"
    if not args.cranelift_only and dynasm.exists():
        runners.append(("dynasm", [str(dynasm)]))
    if not args.dynasm_only and cranelift.exists():
        runners.append(("cranelift", [str(cranelift)]))
    return runners


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynasm-only", action="store_true")
    parser.add_argument("--cranelift-only", action="store_true")
    parser.add_argument("--cpython-only", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="run the whole vendored tree, not just ENABLED")
    parser.add_argument("--filter", default=None,
                        help="run only files whose name contains this substring")
    parser.add_argument("--timeout", type=int, default=120,
                        help="per-file timeout in seconds (default 120)")
    parser.add_argument("--list", action="store_true",
                        help="list files that would run, then exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print pass/fail for every (file, backend)")
    args = parser.parse_args()

    files, off_platform = _files(args)
    if args.list:
        for p in files:
            print(p.name)
        return 0

    runners = _runners(args)
    if not files:
        if off_platform:
            print(f"nothing to run on {os.name}: "
                  f"{off_platform} enabled file(s) are other-platform")
            return 0
        print("no vendored extra_tests files selected", file=sys.stderr)
        return 1
    if not runners:
        print("no runners enabled", file=sys.stderr)
        return 1

    print(f"runners: {[name for name, _ in runners]}")
    print(f"files:   {len(files)}"
          + (f" ({off_platform} skipped on {os.name})" if off_platform else ""))
    print()

    fails: list[tuple[str, str, str]] = []
    passes: dict[str, int] = {name: 0 for name, _ in runners}

    for path in files:
        row = [f"  {path.name:<34s}"]
        row_failed = False
        for backend, cmd in runners:
            ok, detail = _run(cmd, path, args.timeout)
            row.append(f"{backend}={'OK' if ok else 'FAIL'}")
            if ok:
                passes[backend] += 1
            else:
                row_failed = True
                fails.append((path.name, backend, detail))
        if args.verbose or row_failed:
            print(" ".join(row))

    print()
    print("Summary:")
    for backend, _ in runners:
        print(f"  {backend:<10s}  {passes[backend]:>3d}/{len(files)} passed")
    for name, backend, detail in fails:
        print(f"  FAIL {name} [{backend}] {detail}")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
