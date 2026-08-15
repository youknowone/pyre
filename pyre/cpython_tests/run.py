#!/usr/bin/env python3
"""Run the vendored CPython regression suite against a pyre binary.

This mirrors PyPy's `lib-python/conftest.py` mechanism: each CPython test
module is run in its OWN subprocess of the built interpreter, the vendored
test files carry PyPy's own modifications and nothing else, and every
pyre-specific expected-status / skip decision lives in an EXTERNAL baseline
(`baseline.json`) — never as an edit to the test files.

Each module is launched one of three ways, selectable with `--mode`:

  * script (default): `pyre <path>/test_xxx.py` — runs the test file directly
    as `__main__` so its `if __name__ == "__main__": unittest.main()` block
    fires. Needs only `unittest` plus the module's own imports, so it remains
    the most robust mode today. Runner metadata gives
    resource-heavy or dotted-identity-sensitive modules the corresponding
    libregrtest bootstrap without changing this default.
  * module: `pyre -m test.<module>` — same unittest entry but via `runpy`
    and the canonical dotted module identity.
  * regrtest: `pyre -m test -v <module>` — CPython's own libregrtest, the
    PyPy/RustPython form. Libregrtest and `test_regrtest` run cleanly.

A (module, backend) result is classified as:

  PASS        rc 0 (unittest printed OK)
  FAIL        clean nonzero exit (test failures/errors)
  CRASH       rust panic / nonzero internal_compile_panics / signal
  TIMEOUT     exceeded --timeout
  IMPORTERROR module could not even be imported (an interpreter/stdlib gap;
              this is the Phase-0 backlog the suite self-populates)

Gating (default): the run executes only the baseline-PASS subset — the
non-PASS modules carry no gate signal and running the whole suite at the
per-module timeout overruns the CI budget. A module recorded PASS that no
longer passes is a REGRESSION and makes the run exit nonzero. Newly-passing
modules surface only when the non-PASS modules are actually run, so use
`--strict-baseline` (gates on unrecorded improvements), `--full`, or
`--update-baseline` to detect them. `SKIP` baseline entries are not run.

The baseline is a shared file plus a per-host overlay,
`baseline.<sys.platform>-<machine>.json`, consulted first. Whichever host
records a (module, backend) first sets the shared verdict; every other host
writes an entry only where it disagrees, and drops it again once the two
agree. Two separate mechanisms sit next to this: `PLATFORM_GATED` excludes
modules CPython skips wholesale on a host, and `KNOWN_SKIPS` is a decision
about the module rather than the host, so it stays shared.

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
import platform
import signal
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
HOST_TAG = f"{sys.platform}-{platform.machine()}"

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
    "test.test_xpickle": "spawns per-version pythonX.Y subprocesses (PATH-dependent, can deadlock)",
    "test.test_c_locale_coercion": "asserts child stderr is empty, but MAJIT_STATS=1 prints a [jit-stats] line to every process's stderr",
    # `test/test_interpreters/__init__.py` raises SkipTest("GIL disabled")
    # when `test.support.Py_GIL_DISABLED` is set, and pyre reports
    # `Py_GIL_DISABLED = 1` (abiflags `t`) on every host. The package therefore
    # opts out everywhere, which is a fact about the build rather than about
    # one host.
    "test.test_interpreters": "package skips itself on a free-threaded build (Py_GIL_DISABLED)",
}

# Whole modules that CPython skips at import on some platforms. Off-platform
# modules are neither run nor written to the baseline, avoiding false
# `PASS -> SKIP` regressions. Keep each predicate aligned with CPython's guard;
# modules that only skip individual cases do not belong here. Baseline host
# enforcement remains `check.py`'s job (`CPYTHON_SUITE_BASELINE_HOST`).
PLATFORM_GATED = {
    # `if not is_apple: raise unittest.SkipTest("Apple-specific")`, where
    # `test.support.is_apple` is darwin plus the mobile Apple platforms.
    "test.test_apple": (
        lambda p: p in ("darwin", "ios", "tvos", "watchos"),
        "Apple-specific",
    ),
    # `if sys.platform[:3] == 'win': raise unittest.SkipTest(...)`
    "test.test_threadsignals": (
        lambda p: p[:3] != "win",
        "signals not testable on Windows",
    ),
    # `if support.is_emscripten or support.is_wasi: raise unittest.SkipTest(...)`
    "test.test_selectors": (
        lambda p: p not in ("emscripten", "wasi"),
        "cannot create socketpair on Emscripten/WASI",
    ),
    # `if support.is_wasi: raise unittest.SkipTest(...)`
    "test.test_urllib_response": (
        lambda p: p != "wasi",
        "cannot create socket on WASI",
    ),
}


def platform_gate(module: str) -> str | None:
    """The reason `module` does not apply to this host, or `None` if it does."""
    gate = PLATFORM_GATED.get(module)
    if gate is None:
        return None
    applies, reason = gate
    return None if applies(sys.platform) else reason


# These modules intentionally assert on a child interpreter's exact stderr.
# MAJIT_STATS is runner instrumentation rather than language-visible output,
# so do not inject it into those modules or any subprocesses they spawn.
STATS_STDERR_INCOMPATIBLE = {
    "test.test_inspect",
    # test_regrtest exercises both line-oriented `--list-*` output and the
    # worker JSON protocol through several generations of child interpreters.
    # Its subprocess helpers intentionally sanitize their environments, so the
    # root-only ancestry marker cannot keep MAJIT_STATS out of every child.
    "test.test_regrtest",
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
        lines = stderr.splitlines()
        for idx, line in enumerate(lines):
            if "panicked" in line:
                reason = f"rust panic: {line.strip()[:200]}"
                # Rust's default hook prints the panic MESSAGE on the line(s)
                # after 'panicked at file:line:col:'. That body carries the
                # actionable detail; the location line alone cannot triage a
                # crash from a CI log, so append the first message line.
                for follow in lines[idx + 1:]:
                    follow = follow.strip()
                    if follow:
                        reason += f" | {follow[:200]}"
                        break
                return reason
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


def last_stderr_line(err: str) -> str:
    """The last non-empty stderr line — the only evidence a dying run leaves."""
    for line in reversed(err.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def traceback_verdict(lines: list[str], header: int) -> str:
    """The exception line closing the traceback unittest printed at `header`.

    A block runs from the `FAIL:`/`ERROR:` header to the `====` rule opening
    the next one (or unittest's closing `Ran N tests`). Every frame inside a
    traceback is indented, so the exception is the first unindented line after
    the `Traceback` banner. The banner is what arms the search rather than the
    exception simply being the block's last unindented line, because an
    assertion failure prints its diff below the `AssertionError` and those
    lines are unindented too.

    A chained exception prints one traceback per link, joined by `The above
    exception ...` / `During handling ...`. The last link is the one the test
    actually failed with, so each banner overwrites the previous answer.
    """
    verdict = "(no traceback)"
    armed = False
    for line in lines[header + 1:]:
        stripped = line.strip()
        if stripped.startswith("====") or stripped.startswith("Ran "):
            break
        if stripped.startswith("Traceback ("):
            armed = True
            continue
        # Continuation lines of a traceback are indented, its separator rules
        # are punctuation, and blank lines end nothing.
        if not stripped or line[:1].isspace() or set(stripped) <= {"-", "="}:
            continue
        if stripped.startswith("File "):
            continue
        if armed:
            verdict = stripped[:160]
            armed = False
        elif verdict == "(no traceback)":
            # No banner opened this block — a bare `SkipTest` reason, say.
            verdict = stripped[:160]
    return verdict


def failure_digest(out: str, err: str) -> str:
    """Which cases unittest reported against, and its closing verdict.

    A FAIL's tail is not its cause: this runner sets `MAJIT_STATS`, so the last
    thing every process writes is the JIT summary, and `last_stderr_line` reads
    back `Compilation time: 8.1ms` for a module whose real answer is two named
    assertions several hundred lines above it. Every FAIL in the suite recorded
    that same line, which named no test at all.

    unittest writes `FAIL: <case>` / `ERROR: <case>` headers and one closing
    `FAILED (...)`; those are the runner's own account of what went wrong, and
    they are what a CI log needs to carry.

    The header names the case but not the cause, and a case that only fails on
    the CI host cannot be re-run locally to find out — so each header carries
    the closing line of its traceback (the exception type and message), which
    is the one line that says what actually went wrong.
    """
    lines = f"{out}\n{err}".splitlines()
    cases: list[str] = []
    verdict = ""
    for idx, line in enumerate(lines):
        line = line.strip()
        if line.startswith(("FAIL: ", "ERROR: ")):
            entry = f"{line} -> {traceback_verdict(lines, idx)}"
            if entry not in cases:
                cases.append(entry)
        elif line.startswith("FAILED ("):
            verdict = line
    shown = cases[:4]
    if len(cases) > len(shown):
        shown.append(f"(+{len(cases) - len(shown)} more)")
    return " | ".join(part for part in (verdict, *shown) if part)


def death_signal(rc: int) -> str:
    """`SIGBUS` for a run killed by a signal, else a bare return code.

    POSIX `subprocess` reports a signal death as a negative return code; a
    shell-mediated one arrives as 128+n. Naming the signal is what separates
    "the interpreter faulted" from "the interpreter aborted an assertion",
    and those two want completely different investigations.
    """
    num = -rc if rc < 0 else rc - 128
    try:
        return signal.Signals(num).name
    except ValueError:
        return f"rc={rc}"


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
        # Keep the stderr tail. A signal death is precisely the case where no
        # test framework got to report anything, so the last line the
        # interpreter wrote is all the evidence there is — and dropping it
        # left CI runs that could only be re-run, never diagnosed.
        detail = f"signal/abort {death_signal(rc)} rc={rc} {last_stderr_line(err)}"
        return "CRASH", detail.strip()[:200]
    if rc == 0:
        denied = next(
            (
                line[len(RESOURCE_DENIED_MARKER):]
                for line in out.splitlines()
                if line.startswith(RESOURCE_DENIED_MARKER)
            ),
            None,
        )
        if denied is not None:
            return "SKIP", denied[:120]
        return "PASS", ""
    last = last_stderr_line(err)
    ran = "Ran " in out or "Ran " in err or "FAILED (" in out or "FAILED (" in err
    # A module that bails with unittest.SkipTest before any test runs is
    # opting out (a missing optional C extension or wrong platform) — CPython
    # skips it too, so it is honestly SKIP, not an interpreter gap.
    if not ran and "SkipTest" in err:
        return "SKIP", f"rc={rc} {last}"[:120]
    if ran:
        # An IMPORTERROR never reached unittest, so its tail is all there is;
        # a FAIL has unittest's own account, and only that names a test.
        # Wide enough for four cases that each now carry their exception line;
        # at 300 the digest was cut off inside the first case's name.
        return "FAIL", f"rc={rc} {failure_digest(out, err) or last}"[:900]
    return "IMPORTERROR", f"rc={rc} {last}"[:120]


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


def is_package(module: str) -> bool:
    """A `test.<name>` whose `<name>` is a package directory, not a file."""
    name = module.split(".", 1)[1]
    return (TESTDIR / name / "__init__.py").exists()


# Driver that imports a test *package* with real package context (so its
# `__init__.py` / `load_tests` discovers the sub-suite) and runs it through
# unittest, emitting the `Ran N`/`OK`/`FAILED (` markers classify() keys on.
#
# The `if __name__ == "__main__"` guard is the "Safe importing of main module"
# idiom multiprocessing requires: a spawn-started child re-imports this file as
# `__mp_main__` (`multiprocessing/spawn.py:_fixup_main_from_path`), and without
# the guard it would re-run the whole suite inside the child. Every test file
# run directly carries the same guard.
_DOTTED_DRIVER = (
    "import sys, unittest\n"
    "mod = {module!r}\n"
    'if __name__ == "__main__":\n'
    "    __import__(mod)\n"
    "    unittest.main(module=sys.modules[mod], argv=['pyre'], verbosity=2)\n"
)

# Only the resource drivers below write this line, and `classify` honours it
# only from them. A test module that prints its own "skipped: ..." says nothing
# about the run's outcome, so the marker has to be one nothing else emits.
RESOURCE_DENIED_MARKER = "pyre-cpython-suite: ResourceDenied: "

# Same `__mp_main__` guard as _DOTTED_DRIVER.
_RESOURCE_DRIVER = (
    "import runpy\n"
    "from test import support\n"
    'if __name__ == "__main__":\n'
    "    support.use_resources = {{}}\n"
    "    try:\n"
    "        runpy.run_path({path!r}, run_name='__main__')\n"
    "    except support.ResourceDenied as exc:\n"
    "        print({marker!r} + str(exc))\n"
)

_RESOURCE_MODULE_DRIVER = (
    "import runpy\n"
    "from test import support\n"
    "mod = {module!r}\n"
    'if __name__ == "__main__":\n'
    "    support.use_resources = {{}}\n"
    "    try:\n"
    "        runpy.run_module(mod, run_name='__main__', alter_sys=True)\n"
    "    except support.ResourceDenied as exc:\n"
    "        print({marker!r} + str(exc))\n"
)

# test_descr and test_enum assert fully qualified class/enum names. Running
# their files as __main__ changes those names even on CPython, so preserve the
# dotted identity that libregrtest gives them.  test_regrtest and the thread
# suites repeatedly spawn child interpreters and import/filter their canonical
# `test.test_*` modules; running the parent as a second `__main__` module would
# test a different module identity from both libregrtest and those children.
#
# test_runpy's file as __main__ does not run test_runpy at all — it starts
# libregrtest over the whole suite (measured: 491 modules on CPython 3.14,
# 492 here), so script mode can only ever time out on it.
DOTTED_IDENTITY_MODULES = {
    "test.test_descr",
    "test.test_enum",
    "test.test_regrtest",
    "test.test_runpy",
    "test.test_thread",
    "test.test_threading",
}

# These suites create many child interpreters or threads of their own.  Running
# them inside the module-level worker pool can exhaust process/thread resources
# and turn an otherwise 20-second pass into a five-minute timeout.
SERIAL_MODULES = {
    "test.test_regrtest",
    "test.test_thread",
    "test.test_threading",
}

# Modules whose resource-gated tests are ruinous when test.support.use_resources
# is left as None: test_datetime's load_tests then appends an exhaustive test
# class for every installed system timezone, test_urllibnet performs live
# external HTTP requests, and test_zipimport's Zip64 arms build archives past
# 4 GiB, which cost 290s and over 6 GB of RSS.  libregrtest and PyPy's conftest
# use an empty default resource set; the driver also turns a module-level
# ResourceDenied into a clean skip.
DEFAULT_RESOURCE_MODULES = {
    "test.test_datetime",
    "test.test_urllibnet",
    "test.test_zipimport",
}


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
    # Run in a throwaway cwd so a test writing into '.' never touches the
    # repo (the stdlib is resolved relative to the executable, not cwd).
    with tempfile.TemporaryDirectory(prefix="pyre-cpytest-") as cwd:
        if mode == "script" and is_package(module):
            # A package has no `__main__.py` to run as a bare script (and
            # running a submodule file directly breaks its relative imports);
            # drive it through a synthesized unittest entry instead.
            driver = Path(cwd) / "_pyre_pkg_main.py"
            driver.write_text(_DOTTED_DRIVER.format(module=module), encoding="utf-8")
            cmd = [str(binary), str(driver)]
        elif mode == "script" and module in DOTTED_IDENTITY_MODULES:
            driver = Path(cwd) / "_pyre_dotted_main.py"
            driver.write_text(_DOTTED_DRIVER.format(module=module), encoding="utf-8")
            cmd = [str(binary), str(driver)]
        elif mode == "script" and module in DEFAULT_RESOURCE_MODULES:
            # Match libregrtest's runner-owned default resource policy while
            # retaining direct-file __main__ semantics.
            driver = Path(cwd) / "_pyre_script_main.py"
            driver.write_text(
                _RESOURCE_DRIVER.format(
                    path=str(module_path(module)),
                    marker=RESOURCE_DENIED_MARKER,
                ),
                encoding="utf-8",
            )
            cmd = [str(binary), str(driver)]
        elif mode == "module" and module in DEFAULT_RESOURCE_MODULES:
            # `pyre -m` would otherwise leave use_resources as None, which
            # means every optional resource is enabled. Preserve module entry
            # semantics while applying libregrtest's empty default resource set.
            driver = Path(cwd) / "_pyre_resource_module_main.py"
            driver.write_text(
                _RESOURCE_MODULE_DRIVER.format(
                    module=module,
                    marker=RESOURCE_DENIED_MARKER,
                ),
                encoding="utf-8",
            )
            cmd = [str(binary), str(driver)]
        else:
            cmd = build_cmd(binary, module, mode)
        module_env = env
        if module in STATS_STDERR_INCOMPATIBLE:
            module_env = env.copy()
            module_env.pop("MAJIT_STATS", None)
        if mode == "script" and module == "test.test_regrtest":
            # libregrtest/setup.py:88-96 `setup_process` keeps a non-ASCII
            # environment value present for every test process.  Script mode
            # deliberately bypasses that setup, but test_regrtest verifies the
            # runner-owned invariant itself, so reproduce the same environment
            # rather than grading the runner test outside its contract.
            if module_env is env:
                module_env = env.copy()
            module_env.setdefault("PYTHONREGRTEST_UNICODE_GUARD", "æ")
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, env=module_env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired as expired:
            # A bare "timeout 120s" says only that the module is slow, never
            # which case it stopped in — and unittest's progress dots, the one
            # record of how far it got, were being thrown away here. `text=True`
            # decodes what `communicate()` returns; the timeout path raises with
            # the raw chunks it had joined, which is bytes on POSIX and str on
            # Windows, so both spellings have to be accepted.
            partial = expired.stderr or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", "replace")
            return "TIMEOUT", f"timeout {timeout}s {last_stderr_line(partial)}"[:300]
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


def host_baseline_path(path: Path) -> Path:
    """Per-host overlay beside the shared baseline, `baseline.<host>.json`.

    Same shape as the shared file and consulted first, so a host records only
    the modules it genuinely disagrees with.  The jit-stats baselines overlay
    on `sys.platform` alone; a verdict also has to separate the architectures
    within a platform, because the dynasm backend emits different machine code
    on each and a miscompile is not portable.
    """
    return path.with_name(f"{path.stem}.{HOST_TAG}{path.suffix}")


def expected_status(baseline: dict, overlay: dict, module: str,
                    backend: str) -> str | None:
    """Recorded verdict for `module`, the host overlay winning over the shared
    file.  Within one file `dynasm` stands in for a backend with no entry."""
    for source in (overlay, baseline):
        entry = source.get("modules", {}).get(module)
        if entry is None:
            continue
        status = entry.get(backend) or entry.get("dynasm")
        if status is not None:
            return status
    return None


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

    # The report is box-drawn and the module names it echoes come from the
    # stdlib, so a console whose codepage cannot spell one of those characters
    # used to kill the run with a `UnicodeEncodeError` before it reached a
    # single test — the header alone does it on cp949.  The error paths below
    # spell host paths to stderr, which has the same codepage, so reconfigure
    # both streams.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

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
    overlay_path = host_baseline_path(args.baseline)
    overlay = load_baseline(overlay_path) if overlay_path.exists() else {"modules": {}}
    modules = discover_modules(args.filter)

    if args.list:
        for m in modules:
            print(m)
        print(f"\n{len(modules)} modules", file=sys.stderr)
        return 0

    env = dict(os.environ)
    env["MAJIT_STRICT"] = "1"
    env["MAJIT_STATS"] = "1"
    # Several regression tests spawn sys.executable and assert the child's
    # stdout/stderr exactly. Keep the machine-readable panic counters on the
    # direct pyre process without leaking diagnostic output into its children.
    env["PYRE_MAJIT_STATS_ROOT_ONLY"] = "1"
    env.pop("PYRE_MAJIT_STATS_ANCESTOR", None)
    if args.no_jit:
        env["PYRE_NO_JIT"] = "1"

    # Decide which modules to actually run.
    #
    # The gate only protects modules the baseline records as PASS: a PASS that
    # regresses fails CI. In the plain gate run it therefore runs *only* those
    # PASS modules — running the whole suite at the per-module timeout overruns
    # the CI job budget, and the non-PASS modules carry no gate signal. The
    # exploratory lanes still see everything: `--full` reports the full suite,
    # `--update-baseline` re-records it, and `--strict-baseline` must observe
    # non-PASS modules to flag newly-passing ones.
    gate_pass_only = not (args.full or args.update_baseline or args.strict_baseline)
    to_run: list[str] = []
    skipped: list[str] = []
    off_platform: list[tuple[str, str]] = []
    deselected = 0
    for m in modules:
        # Even full/update runs must not overwrite another platform's result.
        gate_reason = platform_gate(m)
        if gate_reason is not None:
            off_platform.append((m, gate_reason))
            skipped.append(m)
            continue
        exp = expected_status(baseline, overlay, m, args.backend)
        is_skip = (exp == "SKIP") or (m in KNOWN_SKIPS)
        if is_skip and not args.full and not args.update_baseline:
            skipped.append(m)
            continue
        if gate_pass_only and exp != "PASS":
            deselected += 1
            continue
        to_run.append(m)

    print(f"pyre CPython suite — backend={args.backend} mode={args.mode} "
          f"jit={'off' if args.no_jit else 'on'} jobs={args.jobs}")
    print(f"binary: {binary}")
    overlay_count = len(overlay.get("modules", {}))
    print(f"baseline: {args.baseline.name} + "
          + (f"{overlay_path.name} ({overlay_count} host entries)"
             if overlay_count else f"no {HOST_TAG} overlay"))
    for m, reason in off_platform:
        print(f"  off-platform on {sys.platform}: {m} ({reason})")
    extra = f", {deselected} not gated (non-PASS)" if deselected else ""
    print(f"{len(to_run)} to run, {len(skipped)} skipped{extra}, "
          f"timeout={args.timeout}s\n")

    results: dict[str, tuple[str, str]] = {}
    done = 0

    def record_result(module: str, result: tuple[str, str]) -> None:
        nonlocal done
        status, detail = result
        results[module] = (status, detail)
        done += 1
        # Every member of STATUSES needs a mark, SKIP included: a module
        # that runs and bails with SkipTest is classified SKIP (:229),
        # which is a different event from a module deselected before the
        # run, and without its own mark it printed as "?" — the character
        # reserved for a status this table does not know.
        mark = {"PASS": "·", "FAIL": "F", "CRASH": "C", "TIMEOUT": "T",
                "IMPORTERROR": "i", "SKIP": "s"}.get(status, "?")
        sys.stdout.write(mark)
        sys.stdout.flush()
        if done % 80 == 0:
            sys.stdout.write(f" {done}/{len(to_run)}\n")

    parallel_modules = [m for m in to_run if m not in SERIAL_MODULES]
    serial_modules = [m for m in to_run if m in SERIAL_MODULES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {
            pool.submit(run_module, binary, m, args.mode, args.timeout, env): m
            for m in parallel_modules
        }
        for fut in concurrent.futures.as_completed(futs):
            m = futs[fut]
            record_result(m, fut.result())
    for m in serial_modules:
        record_result(m, run_module(binary, m, args.mode, args.timeout, env))
    print()

    # Summary by status.
    counts = {s: 0 for s in STATUSES}
    for status, _ in results.values():
        counts[status] = counts.get(status, 0) + 1
    # `+=`, not `=`: SKIP arrives from two places — modules deselected before
    # the run (`skipped`) and modules that ran and bailed with SkipTest
    # (counted from `results` just above). Assigning dropped the second set,
    # so the buckets did not sum to what was selected: a `--filter ctypes`
    # run reported "2 to run" over "?F" and a summary totalling 1.
    counts["SKIP"] += len(skipped)
    print("\n── summary ──")
    for s in STATUSES:
        print(f"  {s:12s} {counts[s]}")
    # Reconcile rather than trust: the buckets partition the selection, so
    # their sum is derivable and a disagreement means a status escaped the
    # table. Computed, not read off the counters that produced it.
    accounted = sum(counts[s] for s in STATUSES)
    expected = len(to_run) + len(skipped)
    if accounted != expected:
        print(f"  {'UNACCOUNTED':12s} {expected - accounted}  "
              f"(buckets sum to {accounted}, {expected} were selected)")
        # Nonzero, and before the analysis below. `check.py` reads this
        # runner's exit code and never this line, so printing alone hands it a
        # pass over a summary short by exactly this many modules. Returning
        # here also stops `--report` and `--update-baseline` from recording a
        # table that is missing a result.
        return 1

    # Regression / improvement analysis vs baseline.
    regressions: list[str] = []
    improvements: list[str] = []
    for m, (status, detail) in sorted(results.items()):
        exp = expected_status(baseline, overlay, m, args.backend)
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
        written = write_baseline(args.baseline, baseline, overlay, results,
                                 args.backend)
        recorded = sum(1 for s, _ in results.values() if s == "PASS")
        for target in written:
            print(f"\nbaseline written: {target} ({recorded} PASS recorded)")
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


def write_baseline(path: Path, baseline: dict, overlay: dict, results: dict,
                   backend: str) -> list[Path]:
    """Record `results`, splitting them between the shared baseline and this
    host's overlay.  Returns the files actually written.

    A verdict the shared file has never seen establishes the shared answer, so
    whichever host records first sets it and no host is privileged.  After
    that a host writes to its own overlay only where it disagrees, and a run
    that comes back into agreement drops the overlay entry again -- so an
    overlay never outlives the divergence that created it.
    """
    modules = baseline.setdefault("modules", {})
    overlay_modules = overlay.setdefault("modules", {})
    baseline["stdlib_version"] = stdlib_version()
    overlay_dirty = False
    for m, (status, _detail) in results.items():
        # Defensive: never overwrite another platform's recorded result.
        if platform_gate(m) is not None:
            continue
        # A curated KNOWN_SKIP stays SKIP regardless of what the run observed
        # (it is a "do not run" decision, not a result). Modules absent from
        # `results` (phantom skips that no longer exist) are simply not added.
        # It is a decision about the module rather than about this host, so it
        # is shared and never overlaid.
        if m in KNOWN_SKIPS:
            entry = modules.setdefault(m, {})
            entry[backend] = "SKIP"
            entry.setdefault("reason", KNOWN_SKIPS[m])
            overlay_dirty |= overlay_modules.pop(m, None) is not None
            continue
        shared = modules.get(m, {}).get(backend)
        if shared is None:
            modules.setdefault(m, {})[backend] = status
            continue
        if status == shared:
            host_entry = overlay_modules.get(m)
            if host_entry is not None and host_entry.pop(backend, None) is not None:
                overlay_dirty = True
                # `reason` is prose about the divergence, so it cannot keep the
                # entry alive once no backend still records one.
                if not any(k != "reason" for k in host_entry):
                    del overlay_modules[m]
            continue
        if overlay_modules.setdefault(m, {}).get(backend) != status:
            overlay_modules[m][backend] = status
            overlay_dirty = True

    written = [path]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    overlay_path = host_baseline_path(path)
    if overlay_dirty or overlay_path.exists():
        if overlay_modules:
            overlay["host"] = HOST_TAG
            overlay["stdlib_version"] = stdlib_version()
            overlay_path.write_text(
                json.dumps(overlay, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            written.append(overlay_path)
        elif overlay_path.exists():
            overlay_path.unlink()
            written.append(overlay_path)
    return written


if __name__ == "__main__":
    sys.exit(main())
