#!/usr/bin/env python3
"""Run the vendored CPython regression suite against a pyre binary.

This mirrors PyPy's `lib-python/conftest.py` mechanism: each CPython test
module is run in its OWN subprocess of the built interpreter, the vendored
test files stay pristine, and every expected-status / skip decision lives in
an EXTERNAL baseline (`baseline.json`) — never as edits to the test files.

Each module is launched one of three ways, selectable with `--mode`:

  * script (default): `pyre <path>/test_xxx.py` — runs the test file directly
    as `__main__` so its `if __name__ == "__main__": unittest.main()` block
    fires. Needs only `unittest` plus the module's own imports; bypasses
    `runpy` / `importlib.util.find_spec` (which pyre's native importer does
    not yet feed), so it is the most robust mode today.
  * module: `pyre -m test.<module>` — same unittest entry but via `runpy`
    (currently blocked until `importlib.util.find_spec` is wired to pyre's
    importer).
  * regrtest: `pyre -m test -v <module>` — CPython's own libregrtest, the
    PyPy/RustPython form (used once libregrtest imports cleanly).

A (module, backend) result is classified as:

  PASS        rc 0 (unittest printed OK)
  FAIL        clean nonzero exit (test failures/errors)
  CRASH       rust panic / nonzero internal_compile_panics / signal
  TIMEOUT     exceeded --timeout
  IMPORTERROR module could not even be imported (an interpreter/stdlib gap;
              this is the Phase-0 backlog the suite self-populates)

Gating (default): a module recorded PASS in the baseline that no longer
passes is a REGRESSION and makes the run exit nonzero. A module that newly
passes is reported as an improvement but does not fail the run (run with
`--update-baseline` to record it). `SKIP` baseline entries are not run.

Usage:
    python3 pyre/cpython_tests/run.py [--backend dynasm|cranelift]
        [--no-jit] [--mode script|module|regrtest] [--jobs N]
        [--timeout SECONDS] [--filter SUBSTR] [--list]
        [--baseline PATH] [--update-baseline] [--strict-baseline] [--full]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TARGET_RELEASE = ROOT / "target" / "release"
TESTDIR = ROOT / "lib-python" / "3" / "test"
STDLIB_VERSION_FILE = ROOT / "lib-python" / "stdlib-version.txt"
DEFAULT_BASELINE = HERE / "baseline.json"

EXE = ".exe" if sys.platform == "win32" else ""
BIN_NAME = {"dynasm": "pyre-dynasm", "cranelift": "pyre-cranelift"}

STATUSES = ("PASS", "FAIL", "CRASH", "TIMEOUT", "IMPORTERROR", "SKIP")

# Modules deliberately not run, ported from PyPy's lib-python/conftest.py
# `testmap` skip reasons. These are CPython implementation-detail or
# environment-specific tests that are never expected to pass on an
# alternative interpreter; keep them out of the gate regardless of state.
KNOWN_SKIPS = {
    "test.test_dis": "implementation detail",
    "test.test_dict_version": "implementation detail",
    "test.test_gc": "implementation detail",
    "test.test_frozen": "implementation detail",
    "test.test_frame": "implementation detail",
    "test.test_sys_setprofile": "implementation detail",
    "test.test_sys_settrace": "implementation detail",
    "test.test_bytecode_helper": "implementation detail",
    "test.test_code": "implementation detail",
    "test.test_peepholer": "implementation detail",
    "test.test_opcache": "implementation detail",
    "test.test_refcounting": "implementation detail (refcount semantics)",
    "test.test_capi": "needs C-API",
    "test.test_cppext": "needs C compiler / C-API",
    "test.test_cext": "needs C compiler / C-API",
    "test.test_stable_abi_ctypes": "needs ctypes.pythonapi",
    "test.test_ossaudiodev": "needs low level audio",
    "test.test_winsound": "needs audio hardware",
    "test.test_tk": "needs display",
    "test.test_ttk": "needs display",
    "test.test_idle": "needs display",
    "test.test_tools": "CPython internal build details",
    "test.test_zipfile64": "demands too many resources",
    "test.test_largefile": "demands too many resources",
    "test.test_embed": "needs embedded CPython",
}

# ── classification ───────────────────────────────────────────────────

def jit_panic_reason(stderr: str) -> str | None:
    """A JIT-level rust panic or nonzero internal_compile_panics, else None.

    Mirrors check.py `_jit_panic_reason`: pyre's panic hook suppresses
    legitimate trace-abandon panics, so anything here is a real crash.
    """
    if not stderr:
        return None
    if "panicked" in stderr:
        for line in stderr.splitlines():
            if "panicked" in line:
                return f"rust panic: {line.strip()[:80]}"
        return "rust panic"
    for line in stderr.splitlines():
        if line.startswith("[jit-stats]") and "internal_compile_panics=" in line:
            field = line.split("internal_compile_panics=", 1)[1].split()[0]
            try:
                if int(field) > 0:
                    return f"internal_compile_panics={field}"
            except ValueError:
                pass
    return None


def classify(rc: int, out: str, err: str) -> tuple[str, str]:
    """Map a finished run to (status, detail).

    The FAIL vs IMPORTERROR split keys on whether unittest actually ran:
    a `Ran N tests` / `FAILED (` line means the module imported and the test
    framework executed (a genuine test FAIL), while its absence means the
    module could not even be loaded — an interpreter/stdlib gap, which is the
    Phase-0 backlog this suite is meant to surface.
    """
    panic = jit_panic_reason(err)
    if panic:
        return "CRASH", panic
    if rc < 0 or rc > 128:
        return "CRASH", f"signal/abort rc={rc}"
    if rc == 0:
        return "PASS", ""
    last = ""
    for line in reversed(err.splitlines()):
        if line.strip():
            last = line.strip()
            break
    ran = "Ran " in out or "Ran " in err or "FAILED (" in out or "FAILED (" in err
    status = "FAIL" if ran else "IMPORTERROR"
    return status, f"rc={rc} {last}"[:120]


# ── discovery ────────────────────────────────────────────────────────

def discover_modules(filter_substring: str | None) -> list[str]:
    """Dotted names (`test.test_xxx`) for every CPython test module/package."""
    names: list[str] = []
    for p in sorted(TESTDIR.glob("test_*.py")):
        names.append(f"test.{p.stem}")
    for d in sorted(TESTDIR.glob("test_*")):
        if d.is_dir() and (d / "__init__.py").exists():
            names.append(f"test.{d.name}")
    names = sorted(set(names))
    if filter_substring:
        names = [n for n in names if filter_substring in n]
    return names


# ── execution ────────────────────────────────────────────────────────

def module_path(module: str) -> Path:
    """Filesystem path to run a dotted `test.<name>` module as a script:
    a `test_xxx.py` file, or a package's `__main__.py`."""
    name = module.split(".", 1)[1]
    pyfile = TESTDIR / f"{name}.py"
    if pyfile.exists():
        return pyfile
    return TESTDIR / name / "__main__.py"


def build_cmd(binary: Path, module: str, mode: str) -> list[str]:
    if mode == "regrtest":
        # CPython libregrtest; <module> is the bare basename (test_xxx).
        return [str(binary), "-m", "test", "-v", module.split(".", 1)[1]]
    if mode == "module":
        return [str(binary), "-m", module]
    # script (default): run the file directly as __main__.
    return [str(binary), str(module_path(module))]


def run_module(binary: Path, module: str, mode: str, timeout: int,
               env: dict) -> tuple[str, str]:
    cmd = build_cmd(binary, module, mode)
    # Run in a throwaway cwd so a test writing into '.' never touches the
    # repo (the stdlib is resolved relative to the executable, not cwd).
    with tempfile.TemporaryDirectory(prefix="pyre-cpytest-") as cwd:
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "TIMEOUT", f"timeout {timeout}s"
    return classify(proc.returncode, proc.stdout or "", proc.stderr or "")


# ── baseline ─────────────────────────────────────────────────────────

def stdlib_version() -> str:
    try:
        text = STDLIB_VERSION_FILE.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    for token in text.split():
        if token.startswith("v") and token[1:2].isdigit():
            return token.lstrip("v")
    return "unknown"


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {"stdlib_version": stdlib_version(), "modules": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def expected_status(baseline: dict, module: str, backend: str) -> str | None:
    entry = baseline.get("modules", {}).get(module)
    if entry is None:
        return None
    return entry.get(backend) or entry.get("dynasm")


# ── main ─────────────────────────────────────────────────────────────

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("dynasm", "cranelift"), default="dynasm")
    ap.add_argument("--binary", help="explicit interpreter path (overrides --backend)")
    ap.add_argument("--no-jit", action="store_true", help="set PYRE_NO_JIT=1")
    ap.add_argument("--mode", choices=("script", "module", "regrtest"), default="script")
    ap.add_argument("--jobs", type=positive_int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=positive_int, default=300)
    ap.add_argument("--filter", help="only modules whose dotted name contains this")
    ap.add_argument("--list", action="store_true", help="list selected modules and exit")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite the baseline from observed results (non-gating)")
    ap.add_argument("--strict-baseline", action="store_true",
                    help="also fail on newly-passing modules not recorded as PASS")
    ap.add_argument("--full", action="store_true",
                    help="run every module and only report (never gate, never skip)")
    ap.add_argument("--report", type=Path,
                    help="write the per-module results as JSON to this path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    binary = Path(args.binary) if args.binary else TARGET_RELEASE / f"{BIN_NAME[args.backend]}{EXE}"
    binary = binary.resolve()
    if not args.list and not binary.exists():
        print(f"error: interpreter not found: {binary}", file=sys.stderr)
        print("       build it with: cargo build --release -p pyrex --bin "
              f"{BIN_NAME[args.backend]} --no-default-features --features {args.backend}",
              file=sys.stderr)
        return 2

    if not TESTDIR.is_dir():
        print(f"error: CPython test suite not found: {TESTDIR}", file=sys.stderr)
        return 2

    baseline = load_baseline(args.baseline)
    modules = discover_modules(args.filter)

    if args.list:
        for m in modules:
            print(m)
        print(f"\n{len(modules)} modules", file=sys.stderr)
        return 0

    env = dict(os.environ)
    env["MAJIT_STRICT"] = "1"
    env["MAJIT_STATS"] = "1"
    if args.no_jit:
        env["PYRE_NO_JIT"] = "1"

    # Decide which modules to actually run.
    to_run: list[str] = []
    skipped: list[str] = []
    for m in modules:
        exp = expected_status(baseline, m, args.backend)
        is_skip = (exp == "SKIP") or (m in KNOWN_SKIPS)
        if is_skip and not args.full and not args.update_baseline:
            skipped.append(m)
            continue
        to_run.append(m)

    print(f"pyre CPython suite — backend={args.backend} mode={args.mode} "
          f"jit={'off' if args.no_jit else 'on'} jobs={args.jobs}")
    print(f"binary: {binary}")
    print(f"{len(to_run)} to run, {len(skipped)} skipped, timeout={args.timeout}s\n")

    results: dict[str, tuple[str, str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_module, binary, m, args.mode, args.timeout, env): m
                for m in to_run}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            m = futs[fut]
            status, detail = fut.result()
            results[m] = (status, detail)
            done += 1
            mark = {"PASS": "·", "FAIL": "F", "CRASH": "C",
                    "TIMEOUT": "T", "IMPORTERROR": "i"}.get(status, "?")
            sys.stdout.write(mark)
            sys.stdout.flush()
            if done % 80 == 0:
                sys.stdout.write(f" {done}/{len(to_run)}\n")
    print()

    # Summary by status.
    counts = {s: 0 for s in STATUSES}
    for status, _ in results.values():
        counts[status] = counts.get(status, 0) + 1
    counts["SKIP"] = len(skipped)
    print("\n── summary ──")
    for s in STATUSES:
        print(f"  {s:12s} {counts[s]}")

    # Regression / improvement analysis vs baseline.
    regressions: list[str] = []
    improvements: list[str] = []
    for m, (status, detail) in sorted(results.items()):
        exp = expected_status(baseline, m, args.backend)
        if exp == "PASS" and status != "PASS":
            regressions.append(f"{m}: PASS -> {status}  {detail}")
        elif exp != "PASS" and status == "PASS":
            improvements.append(f"{m}: {exp or 'new'} -> PASS")

    if improvements:
        print(f"\n── improvements ({len(improvements)}) ──")
        for line in improvements:
            print(f"  + {line}")

    if args.report:
        report = {
            "backend": args.backend,
            "mode": args.mode,
            "jit": not args.no_jit,
            "stdlib_version": stdlib_version(),
            "counts": counts,
            "modules": {m: {"status": s, "detail": d}
                        for m, (s, d) in sorted(results.items())},
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(f"\nreport written: {args.report}")

    if args.update_baseline:
        write_baseline(args.baseline, baseline, results, args.backend)
        print(f"\nbaseline written: {args.baseline} "
              f"({sum(1 for s, _ in results.values() if s == 'PASS')} PASS recorded)")
        return 0

    if regressions:
        print(f"\n── REGRESSIONS ({len(regressions)}) ──")
        for line in regressions:
            print(f"  - {line}")
        # `--full` is report-only (the nightly exploratory lanes rely on it);
        # never let a current PASS regression fail an exploratory run.
        if not args.full:
            return 1

    if args.strict_baseline and improvements and not args.full:
        print("\nstrict: newly-passing modules must be recorded "
              "(run --update-baseline)")
        return 1

    print("\nno regressions" if not regressions else "\n(--full: regressions reported, not gated)")
    return 0


def write_baseline(path: Path, baseline: dict, results: dict, backend: str) -> None:
    modules = baseline.setdefault("modules", {})
    baseline["stdlib_version"] = stdlib_version()
    for m, (status, _detail) in results.items():
        entry = modules.setdefault(m, {})
        # A curated KNOWN_SKIP stays SKIP regardless of what the run observed
        # (it is a "do not run" decision, not a result). Modules absent from
        # `results` (phantom skips that no longer exist) are simply not added.
        if m in KNOWN_SKIPS:
            entry[backend] = "SKIP"
            entry.setdefault("reason", KNOWN_SKIPS[m])
        else:
            entry[backend] = status
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
