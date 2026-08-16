#!/usr/bin/env python3
"""Runner for imported and pyre-authored generic extra_tests snippets.

Runs every `*.py` under `pyre/extra_tests/snippets/` against:
  - CPython (the system `python3`),
  - pyre-dynasm (release build),
  - pyre-cranelift (release build, if present).

A snippet passes when the process exits with code 0.  Unlike
`pyre/parity_tests/run.py` we do NOT require a trailing "OK" line —
the original RustPython runner only checks the return code, so we
mirror that.

The interpreter's cwd is set to the snippets directory so the local
`testutils.py` import succeeds.

Usage:
    python3 pyre/extra_tests/run.py [--dynasm-only|--cranelift-only]
                                    [--cpython-only]
                                    [--gated-only]
                                    [--filter SUBSTRING]
                                    [--timeout SECONDS]
                                    [--list]

Exit code is 0 iff every (script, backend) pair passed.  The imported
corpus is not all green, so a bare run is survey material; use
`--filter` to focus on a category.  `--gated-only` selects the
`# pyre-check: gate=1` subset, which is green everywhere and is what CI
blocks on.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNIPPETS = HERE / "snippets"
ROOT = HERE.parent.parent
TARGET_RELEASE = ROOT / "target" / "release"

EXE = ".exe" if sys.platform == "win32" else ""
PLATFORMS_PREFIX = "# pyre-check: platforms="

# A snippet carrying `# pyre-check: gate=1` is green on every backend and is
# expected to stay that way, so CI runs the marked subset (`--gated-only`) as
# a hard gate.  The rest of the corpus is the imported RustPython set, parts
# of which fail today — some only under CPython, which asserts behaviour the
# original suite never ran there.  Marking a snippet is what moves it from
# survey material to something a red run blocks a merge on; do it only once
# the script passes under CPython and both backends.
GATE_PREFIX = "# pyre-check: gate="

# Snippet basenames that are not standalone test files (helpers /
# scaffolding imported by other snippets).  Skip them from the run.
NON_TEST_FILES = {
    "testutils.py",
}


def _scripts(filter_substring: str | None, gated_only: bool = False) -> list[Path]:
    out: list[Path] = []
    for p in sorted(SNIPPETS.glob("*.py")):
        if p.name in NON_TEST_FILES:
            continue
        header = p.read_text(encoding="utf-8").splitlines()[:20]
        marker = next((line for line in header if line.startswith(PLATFORMS_PREFIX)), None)
        if marker is not None:
            platforms = {
                item.strip()
                for item in marker[len(PLATFORMS_PREFIX):].split(",")
            }
            if sys.platform not in platforms:
                continue
        if gated_only:
            gate = next((line for line in header if line.startswith(GATE_PREFIX)), None)
            if gate is None or gate[len(GATE_PREFIX):].strip() != "1":
                continue
        if filter_substring and filter_substring not in p.name:
            continue
        out.append(p)
    return out


def _expects_failure(script: Path) -> bool:
    """An `xfail_`-prefixed snippet asserts the runner notices a failure.

    The imported RustPython corpus carries these as harness self-tests; they
    pass when the interpreter exits non-zero.
    """
    return script.name.startswith("xfail_")


def _run(cmd: list[str], script: Path, timeout: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd + [str(script)],
            cwd=SNIPPETS,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if _expects_failure(script):
        if proc.returncode != 0:
            return True, ""
        return False, "rc=0 but the snippet is expected to fail"
    if proc.returncode == 0:
        return True, ""
    err = proc.stderr.strip().splitlines()
    last_err = err[-1] if err else ""
    return False, f"rc={proc.returncode} {last_err}"


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
    parser.add_argument("--filter", default=None,
                        help="run only scripts whose name contains this substring")
    parser.add_argument("--gated-only", action="store_true",
                        help="run only scripts marked `# pyre-check: gate=1` "
                             "(the set CI blocks on)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="per-script timeout in seconds (default 30)")
    parser.add_argument("--list", action="store_true",
                        help="list scripts that would run, then exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print pass/fail for every (script, backend)")
    args = parser.parse_args()

    scripts = _scripts(args.filter, args.gated_only)
    if args.list:
        for s in scripts:
            print(s.name)
        return 0

    runners = _runners(args)
    if not scripts:
        print("no extra_tests scripts found", file=sys.stderr)
        return 1
    if not runners:
        print("no runners enabled", file=sys.stderr)
        return 1

    print(f"runners: {[name for name, _ in runners]}")
    print(f"scripts: {len(scripts)}")
    if args.gated_only:
        print("filter:  gate=1 only")
    if args.filter:
        print(f"filter:  {args.filter!r}")
    print()

    fails_per_runner: dict[str, list[tuple[str, str]]] = {n: [] for n, _ in runners}
    passes_per_runner: dict[str, int] = {n: 0 for n, _ in runners}

    for script in scripts:
        name = script.name
        row = [f"  {name:<38s}"]
        for backend, cmd in runners:
            ok, detail = _run(cmd, script, args.timeout)
            mark = "OK" if ok else "FAIL"
            row.append(f"{backend}={mark}")
            if ok:
                passes_per_runner[backend] += 1
            else:
                fails_per_runner[backend].append((name, detail))
        if args.verbose or any(part.endswith("=FAIL") for part in row[1:]):
            print(" ".join(row))

    print()
    print("Summary:")
    for backend, _ in runners:
        passed = passes_per_runner[backend]
        total = len(scripts)
        print(f"  {backend:<10s}  {passed:>3d}/{total} passed")

    total_fails = sum(len(v) for v in fails_per_runner.values())
    if total_fails:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
