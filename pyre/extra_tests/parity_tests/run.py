#!/usr/bin/env python3
"""Parity baseline runner.

Runs every script under `pyre/extra_tests/parity_tests/` against:
  - CPython 3.14 (`PYRE_CHECK_PYTHON3`, else `python3.14` on PATH),
  - the pyre-dynasm binary (release build),
  - the pyre-cranelift binary (release build, if present).

A script passes when:
  - the process exits with code 0,
  - the last non-empty stdout line equals "OK".

Any divergence between CPython and a pyre backend is a parity
regression: the runtime has drifted from CPython observable
semantics.

A script whose subject is platform-specific names the platforms it
applies to in its header (`# pyre-check: platforms=linux,darwin`) and is
skipped elsewhere.

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
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
TARGET_RELEASE = ROOT / "target" / "release"

EXE = ".exe" if sys.platform == "win32" else ""

PLATFORMS_PREFIX = "# pyre-check: platforms="

# The version whose observable behaviour these scripts pin, which is the one
# pyre's native modules (`_sre.MAGIC`) and the vendored `lib-python/3` are
# coupled to. An older CPython disagrees with it on error text, dunder surfaces
# and structure sizes — dozens of failures that are not parity failures, and
# that bury the ones that are. Comparing a backend against the wrong reference
# measures nothing, so a run that cannot find the right one stops rather than
# reporting what it found.
CPYTHON_TARGET = (3, 14)


def _runs_here(path: Path) -> bool:
    """Whether this platform is one the script's expectations hold on.

    A script may name the `sys.platform` values it applies to in its header,
    above the docstring, with the reason beside it:

        # pyre-check: platforms=linux,darwin

    Elsewhere it is skipped, because what it pins is platform-specific and the
    reference CPython fails it too — comparing a backend against a failing
    reference measures nothing. A script with no marker runs everywhere.
    """
    with path.open(encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if not line.startswith(PLATFORMS_PREFIX):
                continue
            named = line[len(PLATFORMS_PREFIX):].strip().split(",")
            return sys.platform in [name.strip() for name in named]
    return True


def _scripts() -> tuple[list[Path], list[Path]]:
    """The scripts to run on this platform, and the ones skipped."""
    out = []
    skipped = []
    for p in sorted(HERE.glob("*.py")):
        if p.name == "run.py":
            continue
        (out if _runs_here(p) else skipped).append(p)
    return out, skipped


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
    if ok:
        detail = ""
    elif proc.returncode == 0:
        # The script ran to completion and never announced itself. Reporting
        # this as `rc=0 last=''` reads like an interpreter that produced
        # nothing, and it is reported once per runner, so three of them make a
        # sound fixture that forgot its last line look like a pyre failure.
        detail = f"exited 0 without a final 'OK' line (last non-empty line {last!r})"
    else:
        detail = f"rc={proc.returncode} last={last!r} stderr={err.strip()!r}"
    return ok, detail


PROBE = "import sys; print(sys.version_info[0], sys.version_info[1]); print(sys.executable)"


def _probe(command: str) -> tuple[tuple[int, int], str] | None:
    """What the interpreter reports as its version and its own path.

    `None` when the command did not run at all, which on Windows is what a
    `python3.14` that names an extensionless shim rather than an executable
    does — `shutil.which` finds it and `CreateProcess` cannot start it.
    """
    try:
        proc = subprocess.run(
            [command, "-c", PROBE],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or "").splitlines()
    if len(lines) < 2:
        return None
    try:
        major, minor = lines[0].split()
    except ValueError:
        return None
    return (int(major), int(minor)), lines[1].strip() or command


def _cpython() -> str:
    """The reference interpreter, which has to be the version the scripts pin.

    `PYRE_CHECK_PYTHON3` names it outright; otherwise the version-qualified
    name is tried before the bare ones, because a `python3` on PATH is
    whichever CPython the system happens to ship. What comes back is the path
    the interpreter reports for itself, so the scripts are spawned by a name
    that does not depend on the PATH they inherit.
    """
    named = os.environ.get("PYRE_CHECK_PYTHON3")
    candidates = [named] if named else ["python3.14", "python3", "python"]
    rejected = []
    for candidate in candidates:
        if named is None and shutil.which(candidate) is None:
            continue
        probed = _probe(candidate)
        if probed is None:
            rejected.append(f"  {candidate}: did not run")
            continue
        version, executable = probed
        if version == CPYTHON_TARGET:
            return executable
        rejected.append("  %s: %d.%d" % (candidate, *version))
    wanted = "%d.%d" % CPYTHON_TARGET
    raise SystemExit(
        f"no CPython {wanted} to compare against — the parity scripts pin its "
        f"behaviour, and an older one fails them for reasons that are not "
        f"parity failures.\n"
        + ("\n".join(rejected) or "  (no candidate on PATH)")
        + "\nName one with PYRE_CHECK_PYTHON3."
    )


def _runners(
    only_dynasm: bool, only_cranelift: bool, gc_poison: bool
) -> list[tuple[str, list[str], dict[str, str] | None]]:
    runners: list[tuple[str, list[str], dict[str, str] | None]] = []
    runners.append(("cpython", [_cpython()], None))
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
    scripts, skipped = _scripts()
    if not scripts:
        print("no parity test scripts found", file=sys.stderr)
        return 1

    print(f"runners: {[name for name, _, _ in runners]}")
    print(f"scripts: {len(scripts)}")
    for script in skipped:
        print(f"skipped ({sys.platform} not in its platforms): {script.name}")
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
