#!/usr/bin/env python3
"""pyre pre-merge check: correctness + regression guard + comparison

Cross-platform Python translation of pyre/check.sh.
"""

import argparse
import difflib
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EXE = ".exe" if sys.platform == "win32" else ""
# pyre's compat target is CPython 3.14; its native modules (`_sre.MAGIC`) and the
# vendored `lib-python/3` are coupled to that version. Take a version-matched
# oracle or none at all: a stale `python3` on PATH (an older system CPython)
# diverges from pypy on version-sensitive error text and trips a spurious
# cpython-vs-pypy baseline mismatch, and its timings gate the ratios.
CPYTHON_TARGET = (3, 14)


def _probe_interpreter(command):
    """What the interpreter reports as its version and its own path.

    `None` when the command did not run at all, which on Windows is what a
    `python3.14` naming an extensionless shim rather than an executable does —
    `shutil.which` finds it and `CreateProcess` cannot start it.
    """
    probe = "import sys; print(sys.version_info[0], sys.version_info[1]); print(sys.executable)"
    try:
        proc = subprocess.run(
            [command, "-c", probe], capture_output=True, text=True, timeout=30,
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


def _resolve_python3():
    """The oracle interpreter, as the absolute path it reports for itself.

    A bare name would be resolved against the PATH of whichever environment
    spawns it, and the timed runs hand the child a curated one.
    """
    named = os.environ.get("PYRE_CHECK_PYTHON3")
    candidates = [named] if named else ["python3.14", "python3", "python"]
    rejected = []
    for candidate in candidates:
        if named is None and shutil.which(candidate) is None:
            continue
        probed = _probe_interpreter(candidate)
        if probed is None:
            rejected.append(f"  {candidate}: did not run")
            continue
        version, executable = probed
        if version == CPYTHON_TARGET:
            return executable
        rejected.append("  %s: %d.%d" % (candidate, *version))
    wanted = "%d.%d" % CPYTHON_TARGET
    raise SystemExit(
        f"no CPython {wanted} to measure against — pyre targets it, and an "
        f"older one disagrees on version-sensitive behaviour and timings.\n"
        + ("\n".join(rejected) or "  (no candidate on PATH)")
        + "\nName one with PYRE_CHECK_PYTHON3."
    )


PYTHON3 = _resolve_python3()
PYPY3 = os.environ.get("PYRE_CHECK_PYPY3") or (
    "pypy3" if shutil.which("pypy3") else "pypy"
)


def _detect_pyre_stdlib():
    """Stdlib directory to pin pyre to.

    pyre's native modules (`_sre`, ...) are coupled to one CPython version
    via `_sre.MAGIC`; the vendored `lib-python/3` is the version-matched
    copy. pyre's own `detect_stdlib_path` already resolves it through
    `find_intree_stdlib`, but a host `python3` whose `re`/`_sre` MAGIC
    disagrees (e.g. an older CPython on the dev PATH, or a PyPy that
    shadows CPython on the CI PATH) would mismatch if reached. Pin the
    vendored copy explicitly so the result is independent of PATH.
    The host `python3` stdlib is only a last resort for an out-of-tree run.
    """
    intree = Path(__file__).resolve().parent.parent / "lib-python" / "3"
    if intree.is_dir():
        return str(intree)
    try:
        proc = subprocess.run(
            [PYTHON3, "-c", "import sysconfig; print(sysconfig.get_paths()['stdlib'])"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    path = (proc.stdout or "").strip()
    return path if path and os.path.isdir(path) else None


PYRE_STDLIB = _detect_pyre_stdlib()

# Which wasm runtime the `pyre-wasm-runner` uses (`--wasm-engine`). wasmtime
# (cranelift) is fast in steady state but recompiles the ~14MB module on every
# process start; wasmi is a pure-Rust interpreter with near-zero startup cost
# but slower hot loops. Forwarded to the runner via PYRE_WASM_ENGINE; ignored
# by the dynasm/cranelift backends.
WASM_ENGINE = "wasmtime"

# The wasm backend has no perf-ratio gate (its run_bench vs_cpython/vs_pypy are
# None) — its per-bench timeout is only a hang guard. wasm legitimately runs a
# few× slower than the native backends the base timeouts were tuned for, and
# slower still on loaded CI runners, so give it extra headroom by default to
# avoid flaky timeouts. Overridable with --wasm-timeout-scale.
WASM_TIMEOUT_SCALE = 4.0
# Ceiling on wasm's execution-only user-CPU as a multiple of dynasm's on the
# same fixture, both startup-subtracted. Unlike the vs-cpython / vs-pypy gates
# this baseline is not a per-bench tuned constant: it is dynasm's own time from
# the same invocation, so the pair sees one host under one load and there is no
# recorded number that can age. Gated only when dynasm actually ran the fixture
# in this run and its execution-only time clears FLOOR_GATE_MIN_BASELINE_S;
# below that the denominator is startup noise, and the comparison table marks
# the ratio `~` the way the pypy gates already do.
#
# 4x rather than the 3x this started at, because 3x was not where the backend
# is. Four fixtures had already been carved out of it one at a time -- fannkuch
# 3.6, fib_recursive 3.6, raise_catch_loop 3.8, short_circuit_value_kept_stack
# 3.7 -- every one for the same structural reason (`wasm_ratio_gate` below
# spells it out), and more kept arriving: one main run failed
# `if_else_jump_forward` at 3.9x and `loop_callee_shared_mutation` at 3.3x with
# no allowance written for either. A ceiling that the fixtures above it are
# exempted from individually is collecting exemptions rather than measuring the
# backend. At 4x the gate means what it says, all four of those allowances come
# out, and one stays genuinely above it
# (`global_quasiimmut_invalidation`, 6).
WASM_MAX_DYNASM_RATIO = 4.0
# Native Windows CI can spend substantially more wall time than reported
# process user-CPU while antivirus and concurrent matrix jobs contend for the
# runner.  Keep the timeout as a hang guard by granting native backends 2x
# headroom; performance still gates on measured user-CPU below.
WIN_NATIVE_TIMEOUT_SCALE = 2.0

BENCH_DIR = "pyre/bench"
SYNTHETIC_BENCH_DIR = "pyre/bench/synth"
CPYTHON_SUITE_BASELINE = "pyre/cpython_tests/baseline.json"
# Per module, matching the CI job. `test.test_asyncio` alone needs ~2m47s of
# wall time, so a smaller per-module limit turns a slow module into a fake
# regression.
CPYTHON_SUITE_MODULE_TIMEOUT_S = 300
CPYTHON_SUITE_TIMEOUT_S = 2400
SNAP_DIR = "pyre/check.snap"
BENCH_COMPARE_BUFFER_S = 0.005
# Windows process CPU accounting advances in scheduler ticks (normally 1/64 s).
# Keep the execution floor at least one observable tick on that platform.
WIN_TIMER_QUANTUM_S = 1.0 / 64
# Empty-program user-CPU startup is MEASURED per interpreter/backend (median of
# STARTUP_SAMPLES runs) and subtracted from every timed run before ratios and
# perf gates are computed, so a large fixed startup — notably wasmtime
# recompiling the ~14MB module on every process spawn (~0.12s user-CPU) — does
# not inflate the ratio/gate of a short bench. Never hardcode the startup.
# Floored so a bench at/below its own startup (or noise) cannot drive a time
# to <= 0 and blow up a ratio.
#
# The startup is a divisor for every short bench: the baseline's exec time is
# `bench - startup`, so a startup estimate that lands high collapses that
# denominator and inflates every ratio computed against it.  A CI runner
# measured pypy startup at 0.031s where the same job on the same platform had
# measured 0.013s, and three unrelated benches failed their gates in that run
# while their pyre exec times had gone *down*.  Five samples make the median
# robust to two outliers instead of one.
#
# The estimator is the MEDIAN and not the minimum, even though the quantity is a
# fixed per-process cost whose error sources are all one-sided — contention,
# cache pressure and page faults only ever add user CPU.  The minimum does
# recover that cost, but it recovers it for an *idle* machine, and it is
# subtracted from bench runs that were not idle: under load a bench carries an
# inflated startup inside it, and taking the unloaded minimum away leaves that
# inflation behind in `exec`.  Pairing median with median cancels it; pairing an
# idle minimum with a loaded bench does not.  Measured on one CI job, the
# minimum came in 15ms under the median for dynasm and 13ms for cranelift, which
# carried nine short benches whose baseline was pinned to EXEC_TIME_FLOOR_S over
# their gates at once.  The estimate has to come from the same load conditions
# as the run it is subtracted from.
STARTUP_SAMPLES = 5
EXEC_TIME_FLOOR_S = WIN_TIMER_QUANTUM_S if sys.platform == "win32" else 0.005
# A floor failure is only trustworthy when the baseline clears the execution
# floor enough for small relative error: execution time is the difference
# between two independently measured values.
#
# Three times the floor is not enough of a clearance to fail a bench on.  At
# that size the baseline is 3 scheduler ticks of Windows CPU accounting, each
# side of the ratio carries +-1 tick, and a "you got fast" verdict is being
# read off a number with a third of its own magnitude in quantization error.
# One CI run failed `comprehension_accumulators` on windows for reading 0.3x
# against a pypy baseline of 0.05s -- 3.2 ticks -- while the same fixture read
# 1.6x and 2.5x on macos.  Ten times the floor holds that error near a tenth,
# and it leaves the gate armed on 24 fixtures whose pypy time is real work.
FLOOR_GATE_MIN_BASELINE_S = 10 * EXEC_TIME_FLOOR_S
# The pypy performance floor is DERIVED from the ceiling and is never stated
# per bench.  What we want from every bench is parity with pypy, so a
# hand-written floor would assert that a fixture must STAY some number of times
# slower than pypy -- not a thing anyone wants to be true, and one more number
# to re-fit every time the ceiling moves.  The floor is only an instrument for
# reporting that a ceiling has gone stale, so it sits at parity: reaching it is
# the event worth a red run.
PERF_GATE_FLOOR_RATIO = 1.0
# Below a ceiling of this many times parity, a floor AT parity would sit inside
# the band the ceiling itself allows, so the floor drops proportionally instead.
#
# The divisor is what lets ONE number bound a fixture at both ends, so it has to
# clear the span that fixture reads across the runners TWICE OVER: a ceiling is
# set at twice the slowest runner's ratio, and the derived floor still has to
# land under the fastest runner's.  Measured over the runner logs, a fixture's
# own ratio spans as much as 17.8x end to end (`defaults_reassigned_midloop`
# reads 1.0x and 17.8x), and `class_attrs_methods` and
# `foriter_inplace_immutable` are barely narrower -- so a ceiling honestly
# fitted to the slow end sits ~36x above the fast end's reading.  A divisor
# under that turns raising a ceiling into a floor failure on the fast runner,
# which is the one way this pair of bounds can fight itself.
PERF_GATE_FLOOR_DIVISOR = 40
# A single slow sample is retried before failing a performance gate. Windows
# needs more samples because its process CPU accounting is scheduler-tick
# quantized (see WIN_TIMER_QUANTUM_S above).
PERF_RETRY_RUNS = 5 if sys.platform == "win32" else 3
SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S = 5
CARGO_CONFIG = {
    "dynasm": {
        "extra": ["--no-default-features", "--features", "dynasm"],
        "bin": "pyre-dynasm",
    },
    "cranelift": {
        "extra": ["--no-default-features", "--features", "cranelift"],
        "bin": "pyre-cranelift",
    },
    # The wasm backend is not a `pyrex` binary: it is the wasm32 build of
    # `pyre-wasm` (full interpreter+JIT) executed by the native `pyre-wasm-runner`
    # under wasmtime. `build_backend` special-cases it (see `wasm=True`).
    "wasm": {
        "bin": "pyre-wasm-runner",
        "wasm": True,
    },
}

# Module + linker flags for the wasm32 build of `pyre-wasm`:
#   * `--export-table` exposes `__indirect_function_table` so the runner's
#     `jit_call` trampoline can dispatch residual calls by table index;
#   * `getrandom_backend="custom"` selects getrandom's custom backend (see
#     `pyre-wasm/src/lib.rs`) so the module carries no wasm-bindgen imports.
# `pyre-wasm` builds to the same `pyre_wasm.wasm` filename for both the `web`
# and `wasm-host` features, so a later build of the other flavour would clobber
# the native-host module. Copy the wasm-host build to a distinct, stable path the
# runner reads, immune to that overwrite. `pyre/pyre-wasm/build-web.sh` does the
# mirror image for the web flavour (snapshot -> pyre_wasm.web.wasm, fed to
# wasm-bindgen).
WASM_BUILD_OUTPUT = "target/wasm32-unknown-unknown/release/pyre_wasm.wasm"
WASM_MODULE_PATH = "target/wasm32-unknown-unknown/release/pyre_wasm.wasm-host.wasm"
# The JIT's trace-abort signal (InvalidLoop / speculative-fold failure) is
# propagated as a `Result`/deferred flag through the optimizer rather than a
# panic, so the build needs neither unwinding nor `-Z build-std`: it runs on the
# precompiled wasm32 std with the default `panic=abort`, on the stable toolchain.
# `--export-table` exposes the indirect-call table the runner patches for JIT
# re-entry; `--growable-table` drops its fixed maximum so the host can append
# compiled trace functions for inter-trace call_indirect chaining;
# `getrandom_backend="custom"` selects the no-import getrandom backend.
WASM_RUSTFLAGS = (
    '-C link-arg=--export-table -C link-arg=--growable-table '
    '--cfg getrandom_backend="custom"'
)
# Stable toolchain, no build-std. Kept as a (possibly empty) arg list so the
# build invocation can splat it uniformly.
WASM_CARGO_TOOLCHAIN = []
WASM_BUILD_STD_FLAGS = []

# ── ANSI helpers ─────────────────────────────────────────────────────

def red(s):    return f"\033[31m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

# ── Child-process user CPU time and peak RSS ─────────────────────────

# Peak RSS in MiB of the most recent `run_timed` child, or None when the run
# timed out or the platform cannot report it. A single slot rather than a
# return-tuple member: `run_timed`'s 4-tuple has many callers and only the
# memory gate reads this. Runs are sequential, so the slot is unambiguous.
_LAST_RUN_MAXRSS_MB = [None]


def last_run_peak_rss_mb():
    """Peak RSS (MiB) of the child `run_timed` most recently reaped."""
    return _LAST_RUN_MAXRSS_MB[0]



def _run_timed_unix(args, timeout_s, env=None):
    # `os.wait4` rather than `getrusage(RUSAGE_CHILDREN)` around
    # `subprocess.run`: `ru_maxrss` on RUSAGE_CHILDREN is a running maximum
    # over every reaped child, not a counter, so a before/after difference
    # attributes nothing to this run. wait4 reports the rusage of this child
    # alone, which is what a per-fixture memory gate needs. `ru_utime` comes
    # from the same struct, so the timing path gains a per-child number too
    # instead of a difference of process-wide totals.
    #
    # The child's pipes are replaced by temporary files: `communicate()` would
    # reap the child itself and leave nothing for wait4, and draining pipes by
    # hand risks the classic full-buffer deadlock.
    import resource  # noqa: F401  (documents the rusage struct's origin)
    import tempfile
    import threading

    timed_out = False

    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        proc = subprocess.Popen(args, stdout=out_f, stderr=err_f, env=env)

        def _kill():
            nonlocal timed_out
            timed_out = True
            proc.kill()

        timer = threading.Timer(timeout_s, _kill) if timeout_s else None
        if timer is not None:
            timer.start()
        try:
            _pid, status, usage = os.wait4(proc.pid, 0)
        finally:
            if timer is not None:
                timer.cancel()
        # Popen still believes the child is running; tell it otherwise so its
        # finalizer neither waits nor warns.
        proc.returncode = -(status & 0x7F) if status & 0x7F else (status >> 8)

        if timed_out:
            _LAST_RUN_MAXRSS_MB[0] = None
            return "", 0.0, 124, ""

        out_f.seek(0)
        err_f.seek(0)
        stdout_bytes = out_f.read()
        stderr_bytes = err_f.read()

    # ru_maxrss is kilobytes on Linux and bytes on macOS/BSD.
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    _LAST_RUN_MAXRSS_MB[0] = usage.ru_maxrss / scale
    return (
        stdout_bytes.decode("utf-8", errors="replace"),
        max(usage.ru_utime, 0.0),
        proc.returncode,
        stderr_bytes.decode("utf-8", errors="replace"),
    )


def _run_timed_win32(args, timeout_s, env=None):
    import ctypes
    from ctypes import wintypes

    # pypy3.exe on Windows is a launcher that spawns the real interpreter as a
    # child process; GetProcessTimes on Popen's handle only sees the launcher
    # and reports ~0s. A JobObject aggregates user/kernel time across all
    # descendant processes, which is what we actually want.
    class _IOCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JobBasic(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", ctypes.c_uint32),
            ("TotalProcesses", ctypes.c_uint32),
            ("ActiveProcesses", ctypes.c_uint32),
            ("TotalTerminatedProcesses", ctypes.c_uint32),
        ]

    class _JobBasicAndIo(ctypes.Structure):
        _fields_ = [("BasicInfo", _JobBasic), ("IoInfo", _IOCounters)]

    kernel32 = ctypes.windll.kernel32
    job = kernel32.CreateJobObjectW(None, None)

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    # Assigning right after Popen catches launchers like pypy3.exe before they
    # spawn their interpreter child; descendants inherit job membership.
    assigned = bool(kernel32.AssignProcessToJobObject(job, int(proc._handle)))

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        kernel32.CloseHandle(job)
        return "", 0.0, 124, ""

    utime = 0.0
    JobObjectBasicAndIoAccountingInformation = 8
    if assigned:
        info = _JobBasicAndIo()
        if kernel32.QueryInformationJobObject(
            job, JobObjectBasicAndIoAccountingInformation,
            ctypes.byref(info), ctypes.sizeof(info), None,
        ):
            utime = info.BasicInfo.TotalUserTime / 1e7
    else:
        # Job assignment refused (e.g. already in a non-nestable job on older
        # Windows). Fall back to per-process times.
        ct = wintypes.FILETIME()
        et = wintypes.FILETIME()
        kt = wintypes.FILETIME()
        ut = wintypes.FILETIME()
        if kernel32.GetProcessTimes(
            int(proc._handle),
            ctypes.byref(ct), ctypes.byref(et),
            ctypes.byref(kt), ctypes.byref(ut),
        ):
            utime = ((ut.dwHighDateTime << 32) | ut.dwLowDateTime) / 1e7
    kernel32.CloseHandle(job)
    return (
        stdout_bytes.decode("utf-8", errors="replace"),
        utime,
        proc.returncode,
        (stderr_bytes or b"").decode("utf-8", errors="replace"),
    )


def run_timed(args, timeout_s=None, env=None):
    """Run *args*, return (stdout_str, user_cpu_seconds, returncode, stderr_str).

    returncode 124 = timeout (matching coreutils convention). *env* (when
    given) replaces the child environment (pass a full os.environ copy plus
    extras).
    """
    if sys.platform == "win32":
        # No wait4 counterpart; the memory gate reads None and stays inert.
        _LAST_RUN_MAXRSS_MB[0] = None
        out, t, rc, err = _run_timed_win32(args, timeout_s, env)
    else:
        out, t, rc, err = _run_timed_unix(args, timeout_s, env)
    # PyPy/CPython on Windows emit CRLF in stdout text mode; Rust's println!
    # emits LF on all platforms. Normalize so output comparisons aren't
    # platform-sensitive (and snapshots stay portable).
    return out.replace("\r\n", "\n"), t, rc, err


# Names the child is allowed to inherit. Everything else in this process's
# environment is dropped before the child sees it — see `child_env_base`.
ENV_ALLOWLIST = (
    # Locating the loader, the shell a subprocess needs, and scratch space.
    "PATH",
    "PATHEXT",
    "COMSPEC",
    "SHELL",
    "TEMP",
    "TMP",
    "TMPDIR",
    # Windows refuses to start a process, resolve a system DLL or initialise
    # WinSock without these.
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    # The home a `~` expands to, and the account naming it.
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "USER",
    "USERNAME",
    "LOGNAME",
    # Shared-library search, which a build out of the source tree depends on.
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    # The locale chain, because an unset chain resolves to the C locale and
    # changes how the child decodes its own stdio — not something to vary
    # under a run whose stdout is diffed against an oracle.
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PYTHONIOENCODING",
)

# Prefixes a developer sets deliberately to steer this run, kept so an A/B
# stays reachable without editing this file.
ENV_ALLOWLIST_PREFIXES = ("PYRE_", "MAJIT_", "PYPY_", "PYTHON")


def child_env_base():
    """The environment a measured child starts from: an allowlist, not a copy.

    pyre copies every environment entry into `posix.environ` at startup
    (`interp_posix.rs:882 create_environ`, PyPy's `_convertenviron`), so the
    environment's total size is startup allocation like any other. That
    allocation moves where the nursery fills, which moves where a collection
    lands, which changes how many times a compiled trace re-enters a guarded
    branch afterwards — and `guard_failures` counts those re-entries.

    Measured on one binary with nothing but unrelated variables added:
    `str_fstring` read 658 or 659, entirely from one guard in trace 6 firing
    51 times or 52. `loops_compiled` and `bridges_compiled` did not move at
    all, so nothing about what was compiled differed — only how often the
    compiled code re-entered one branch. That fixture no longer demonstrates
    it: its trip counts were halved so it stops short of the major-collection
    threshold entirely, which is the durable fix for one fixture and not for
    the input.

    A CI runner injects dozens of variables a developer's shell does not
    (GITHUB_*, RUNNER_*, JAVA_HOME, and the rest), and those are exactly the
    class of input shown above to move a counter. Dropping them removes that
    input rather than relaxing the comparison. It is not established that it
    is the *only* per-host input — this pin narrows what a counter can depend
    on; it does not prove a counter now depends on nothing but the tree.

    The allowlist is not a fixed environment either: the `PYTHON` prefix keeps
    whatever a host names that way (PYTHONLOCATION, PYTHON3_ROOT_DIR on the
    runners), because the same prefix carries the PYTHONSAFEPATH and
    PYTHONDONTWRITEBYTECODE overrides a developer sets deliberately and that
    `LAUNCH_ENV_NAMES` forwards to the wasm guest. What it removes is the bulk.

    Any non-empty `PYRE_CHECK_INHERIT_ENV` restores the old whole-environment
    copy, so the effect of this normalisation stays measurable.
    """
    if os.environ.get("PYRE_CHECK_INHERIT_ENV"):
        return dict(os.environ)
    env = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper in ENV_ALLOWLIST or upper.startswith(ENV_ALLOWLIST_PREFIXES):
            env[name] = value
    return env


def pyre_env():
    """Child environment for pyre runs: strict JIT plus one-line stats.

    MAJIT_STRICT=1 re-raises internal compile panics instead of silently
    falling back to the interpreter, so a JIT bug surfaces as a crash here
    rather than as correct-but-uncompiled output. MAJIT_STATS=1 prints the
    `[jit-stats]` line that `_jit_panic_reason` inspects.

    PYRE_DESCR_SPELLING_GATE=1 makes the descr-universe counters in that line
    name their members. The counters are gated at zero
    (JITSTATS_BADNESS_FIELDS), and a bare `descr_set_ambiguous 0 -> 1` says
    nothing about which container disagreed — on a platform the developer
    cannot reproduce, the name is the whole diagnosis. Naming costs one stderr
    line per member and only ever prints when a member fails to resolve.
    """
    env = child_env_base()
    env["MAJIT_STRICT"] = "1"
    env["MAJIT_STATS"] = "1"
    env["PYRE_DESCR_SPELLING_GATE"] = "1"
    # Pin the nursery, so the minor-collection schedule is a property of the
    # tree rather than of the machine. `default_nursery_size` (collector.rs)
    # falls back to `estimate_best_nursery_size`, which is half the L2 cache
    # once that exceeds 8MB and `DEFAULT_NURSERY_SIZE` (4MB, env.py's
    # `NURSERY_SIZE_UNKNOWN_CACHE`) otherwise. That probe is macOS-only —
    # `get_l2cache` is `-1` under `cfg(not(target_os = "macos"))` — so two
    # macOS machines can disagree while every Linux and Windows host already
    # sits on the 4MB fallback.
    #
    # `guard_failures` is what moved: it counts bailouts, and a collection
    # landing a little earlier or later moves them. Measured — a macos runner
    # reported 19 differences against baselines recorded here, every one of
    # them `guard_failures` and most of them identical on both backends (so the
    # difference is in the schedule, not in codegen: no `loops_compiled` or
    # `bridges_compiled` moved at all). Sweeping `PYPY_GC_NURSERY` reproduced
    # that runner's numbers exactly at 6MB — `fib_loop` 189, `dict_set` 609,
    # `listcomp_hot` 239, `nested_list_comprehension_hot` 402,
    # `recursive_call_frame_relocation` 636, `str_fstring` 657,
    # `closure_per_call` 415, `type_name_setter` 2, all eight. A 12MB L2 halves
    # to exactly that.
    #
    # 4MB is the documented unknown-cache value and what the non-macOS hosts
    # compute anyway, so pinning it costs nothing there and makes a macOS
    # recorder agree with them. It no longer decides `guard_failures` — the
    # threshold pin below removes the collection those readings were counting,
    # and the baselines are recorded with both pins in place — but it still
    # fixes the minor schedule and the nursery term the threshold derives from.
    env.setdefault("PYPY_GC_NURSERY", str(4 * 1024 * 1024))
    # Pin the major-collection threshold too, because the nursery pin above only
    # fixes one term of it. `min_heap_size` is `PYPY_GC_MIN`, else `nursery * 8`
    # (collector.rs:648), floored by `nursery * major_collection_threshold`
    # (:662), and installed as `next_major_collection_threshold` (:702) — so a
    # 4MB nursery also sets a 32MB threshold, and where old-gen use crosses it
    # (:2437 `threshold_reached`) stayed free to move.
    #
    # Crossing it used to reach `guard_failures`, and no longer does. The major
    # step's finalizer trigger arms the eval-breaker word, and every compiled
    # loop's back edge polls that word through a real guard —
    # `RawLoadI(&EVAL_BREAKER_WORD) -> IntAnd(word, JIT_BREAKER_MASK) ->
    # IntIsTrue -> GuardFalse` (trace_opcode.rs) — whose failure was counted
    # like any other. It is now tallied separately as `back_edge_polls`
    # (`is_back_edge_poll_guard` marks the guard, `record_guard_failure_event`
    # splits the total), which is what removes the whole class rather than
    # avoiding it; the rest of this note is kept because it describes the pins
    # that are still in place and what they were measured to do. The trace is
    # peeled, so that
    # poll appears twice, and which copy catches the armed bit sets the price:
    # the loop-body copy costs one bailout, the peeled-preamble copy costs two,
    # because resuming from it re-enters at the loop head and fails the
    # preamble's FOR_ITER guard once more. That is the whole of the
    # `recursive_call_frame_relocation` 638/637 knife edge — 638 preamble, 637
    # body, 636 no crossing at all — with `loops_compiled` and
    # `bridges_compiled` identical across all three, and it is why the windows
    # runner disagreed with itself between jobs rather than against the tree.
    #
    # Pushing the threshold past a fixture's working set removes the event
    # instead of relocating it. Measured: `recursive_call_frame_relocation`
    # reads 636 at every nursery from 3968KB to 8MB, and `closure_per_call`
    # reads 414 where it had alternated 415/416 — the pair a per-platform
    # overlay could not hold.
    #
    # "Past every fixture's working set" is what this pin was believed to do
    # and is not what it does — the two fixtures above are not the suite.
    # Measured across all 404 dynasm synthetic fixtures, comparing each one's
    # count at this 256MB pin against the same fixture at 8GB (where no major
    # collection happens at all): 11 of them still cross, for 21 poll failures
    # in total. `bytes_split_whitespace_maxsplit` reads 4 here and 1 there,
    # `build_set_hashability` 4 and 1, `str_fstring` 658 and 657. So the pin
    # narrows the class and does not close it, which is why the counter split
    # above exists — those 11 were the fixtures that flipped between hosts.
    #
    # The residual is not more of the same class, though, and calling those 21
    # "poll failures" is wrong. Taking `str_fstring` apart guard by guard: its
    # 658-vs-657 is one guard in trace 6 firing 52 times or 51, and that guard
    # is an ordinary branch, not the breaker poll. The poll is what bails out;
    # re-entry after the bailout takes the branch once more, and that hit is
    # what the counter records. So the counter split above removed the poll
    # class exactly as described, and what is left is a real deopt the split
    # cannot absorb and no GC pin closes, because it is not a crossing being
    # counted. What closes it for a fixture is keeping the fixture short of the
    # crossing — see `str_fstring.py`'s header, where halving the trip counts
    # is what settled this one. Holding the startup allocation still
    # (`child_env_base`) narrows what can move the placement; it does not
    # remove the placement. Host RAM cannot re-open it: `max_delta`
    # (0.125 * total memory) enters only as an upper bound at collector.rs:3570
    # and the `min_heap_size` floor is applied after it (:2452), so the floor
    # wins on every host.
    #
    # "Every fixture's working set" is the criterion, but the value was chosen
    # against the two fixtures that motivated it, and `str_fstring` sits above
    # it. That fixture merges six hot loops into one run, and at 256MB it still
    # crossed: it reads `guard_failures` 658 or 659 as a function of nothing but
    # how much the process allocated before the loops got hot. Startup
    # allocation is the whole input — a 49-variable environment reads 659 and a
    # 9-variable one 658, with the same binary, the same nursery and the same
    # fixture, non-monotonically in the variable count. That is the same
    # preamble-vs-body knife edge described above, and it is what the three
    # per-platform overlays were holding: each host has its own startup
    # allocation volume, so each landed on its own side of the crossing.
    #
    # 512MB clears it. `str_fstring` then reads 657 at every nursery from 4MB to
    # 32MB and under every perturbation of that startup volume — the same
    # invariance that says "no crossing at all" for the two fixtures above.
    # Eight other fixtures move with it, none of them a knife edge at either
    # value: `arith_int_bool` 2214 -> 2212, `build_set_hashability` 4 -> 2,
    # `bytes_split_whitespace_maxsplit` 3 -> 2,
    # `delete_negative_open_slice_hot` 1405 -> 1404, `instance_dict_reassign`
    # 2 -> 1, `newslice_step_hot` 5 -> 3 and `unpack_ex_hot` 3 -> 2 lose a
    # crossing, and `inlined_helper_mutation` 3 -> 4 relocates one. No
    # `loops_compiled` or `bridges_compiled` moves anywhere in the suite, which
    # is the signature that says the collection schedule is the only thing that
    # changed.
    #
    # The cost is memory, and it is paid by the one fixture that was over the
    # old threshold: `str_fstring` peaks at 618MB against 519MB. Every other
    # fixture is flat to within a megabyte (`closure_per_call` 179 -> 180MB,
    # `recursive_call_frame_relocation` 159 -> 160MB), because a threshold only
    # costs what a fixture actually retains.
    #
    # Both pins reach the wasm backend only because `pyre-wasm-runner` hands
    # them to the guest explicitly (`pyre_set_gc_env`): a wasm32-unknown-unknown
    # module's `std::env` is permanently empty, so setting them here does
    # nothing on its own there.
    #
    # Reaching the guest is not the same as clearing the crossing there. 256MB
    # removed it on both native backends and did not in the guest, whose
    # baselines held `recursive_call_frame_relocation` 638 and
    # `closure_per_call` 415 against their 636 and 414 — the same knife edge
    # described above, one step along it. Historically the two ran different
    # interpreter allocation models: `PYRE_GC_INTERP` was off natively and on in
    # the guest, so only the guest counted interpreter-created int/float boxes
    # toward the old-gen threshold. The gate is now default-on everywhere and
    # the allocation model is shared, but the recorded counter boundary outlived
    # that convergence, because the threshold was still inside the guest's
    # working set.
    #
    # The value below is past it, so the guest now lands exactly on the native
    # counts: both fixtures re-recorded to 636 and 414, identical across all
    # three backends. It buys that agreement with memory — `closure_per_call`
    # peaked at 459MB under the guest at 384MB against 358MB at 256MB. The
    # sensitivity this note is here to name outlives the convergence: a change
    # in interpreter-path allocation volume moves the wasm numbers while the
    # native ones sit still.
    env.setdefault("PYPY_GC_MIN", str(512 * 1024 * 1024))
    # Keep the bench directory off `sys.path` (`-P`), so the jit-stats counters
    # describe the fixture rather than the directory it happens to sit in.
    #
    # `sys.path[0]` is the script's own directory, and `pyre/bench/synth` holds
    # well over a thousand entries. The import machinery's scan of that entry is
    # itself a Python loop, and at that size it crosses the 1039 compile
    # threshold — so a fixture with no loop at all in its body still records
    # `loops_compiled=1`, produced by a single `import` statement. Measured, one
    # fixture unchanged and only the directory it runs from varied: 400 files
    # compile nothing, 800 compile that loop, and the real directory is well past
    # both. It is invariant to PYPY_GC_NURSERY (1MB..16MB), so it is not the
    # collection schedule.
    #
    # That is what made the counters disagree between platforms: path handling
    # differs per OS, so the same ambient loop lands either side of the
    # threshold. Every jit-stats diff CI reported was a uniform +-1 on
    # `loops_compiled` — and setting this flag reproduces the Linux numbers
    # exactly on macOS (`arith_int_bool` 8 -> 7 with guard_failures 2213 -> 2211,
    # `gc_deque_backing_list` 6 -> 5 / 204 -> 203, `struct_pack_unpack`
    # 2 -> 1 / 2 -> 1).
    #
    # `-P` alone, not `-I`: the latter also implies `-E`, which drops
    # PYTHONIOENCODING and so changes how the child resolves its stdio encoding —
    # not something to fold into a run whose stdout is diffed against an oracle.
    #
    # An explicit PYTHONSAFEPATH in the environment wins, so the old behaviour
    # stays reachable for an A/B: pyre reads the variable as a presence flag, so
    # passing it empty is what turns the flag back off.
    env.setdefault("PYTHONSAFEPATH", "1")
    # Write no bytecode cache, so a reading describes the tree rather than what
    # earlier runs left on disk. Importing a module the cache already holds
    # unmarshals where importing it cold compiles, and that difference reaches
    # jit-stats the same way the ambient import loop above does: `import json,
    # base64, textwrap` alone drops 30 `__pycache__/*.pyre314.pyc` files into
    # the pinned stdlib, so the first run of a fresh checkout and every run
    # after it measure different work.
    #
    # It also removes a class of platform disagreement rather than papering
    # over it. On Windows nothing pyre wrote was readable until FileIO put an
    # adopted descriptor in binary mode — `_write_atomic` wraps a raw fd, whose
    # CRT text mode turned every `\n` into `\r\n` — so the same tree read one
    # set of counters while the cache was dead and another once it worked.
    #
    # This suppresses writing only; a stale cache from before this pin is still
    # read, so an A/B against an older tree wants `__pycache__` cleared first.
    # `launch_env::finalize` folds the name as an integer flag, so an explicit
    # PYTHONDONTWRITEBYTECODE=0 turns it back off. It reaches the wasm guest
    # too: the name is in `LAUNCH_ENV_NAMES`, which the runner forwards through
    # `pyre_set_launch_env`.
    #
    # Only pyre is pinned this way. The oracles are spawned with the inherited
    # environment (`run_timed([PYTHON3, script])`), so they keep writing and
    # reading their own caches and import warm from their first run onward,
    # while pyre now imports cold on every run. Startup subtraction covers the
    # imports an empty program performs, not the ones a fixture adds on top, so
    # for the handful that `import json` / `re` / `pickle` the ratio carries
    # that difference. Determinism is the trade taken here: a warm pyre depends
    # on what earlier runs left behind, and a ratio that moves with disk state
    # is worse than one that is consistently cold.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Pin the vendored, `_sre.MAGIC`-matched stdlib so pyre never picks up a
    # version-mismatched host `python3` off the PATH. An explicit PYRE_STDLIB
    # in the environment wins.
    if PYRE_STDLIB and "PYRE_STDLIB" not in env:
        env["PYRE_STDLIB"] = PYRE_STDLIB
    # Point the wasm runner at the built module by absolute path so it resolves
    # regardless of the child's working directory (ignored by other backends).
    if "PYRE_WASM_MODULE" not in env and Path(WASM_MODULE_PATH).exists():
        env["PYRE_WASM_MODULE"] = str(Path(WASM_MODULE_PATH).resolve())
    # Pick the wasm runtime engine (ignored by other backends). An explicit
    # PYRE_WASM_ENGINE in the environment wins over the --wasm-engine default.
    if "PYRE_WASM_ENGINE" not in env:
        env["PYRE_WASM_ENGINE"] = WASM_ENGINE
    return env


def _dump_failed_run(output, stderr, limit=40):
    """Print the tail of a failed run's captured streams.

    A self-checking guard reports only its exit code, so without this the
    traceback that explains the failure never reaches the log and a
    platform-specific break has to be reconstructed from the source. Bounded to
    the last *limit* lines of each stream so a runaway script cannot flood CI.
    """
    for label, text in (("stdout", output), ("stderr", stderr)):
        if not text or not text.strip():
            continue
        lines = text.rstrip().splitlines()
        clipped = lines[-limit:]
        print(f"─── {label} ───")
        if len(clipped) < len(lines):
            print(dim(f"... {len(lines) - len(clipped)} earlier line(s) omitted"))
        for line in clipped:
            print(line)
    print("────────────────────")


def _dump_output_mismatch(actual, expected, limit=80):
    """Print a bounded unified diff for a backend stdout mismatch."""
    diff = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="expected (pypy)",
            tofile="actual",
            lineterm="",
        )
    )
    if not diff:
        # Preserve otherwise-invisible trailing-newline/line-ending details.
        print(f"    expected repr: {expected!r}")
        print(f"    actual repr:   {actual!r}")
        return
    print("─── output diff ───")
    for line in diff[:limit]:
        print(line)
    if len(diff) > limit:
        print(dim(f"... {len(diff) - limit} later diff line(s) omitted"))
    print("────────────────────")


def _jit_panic_reason(stderr):
    """Return a failure reason if *stderr* shows a JIT-level Rust panic or a
    nonzero internal_compile_panics stat, else None.

    A Rust panic prints 'panicked at' via the default hook (InvalidLoop is
    suppressed by pyre's panic hook, so legitimate trace aborts never appear
    here). A nonzero internal_compile_panics in the `[jit-stats]` line means an
    internal compile bug fell back to the interpreter (only reachable in a
    non-strict build; under MAJIT_STRICT the panic re-raises and shows up as
    'panicked' plus a nonzero exit instead).
    """
    if not stderr:
        return None
    if "panicked" in stderr:
        lines = stderr.splitlines()
        for idx, line in enumerate(lines):
            if "panicked" in line:
                reason = f"rust panic: {line.strip()[:80]}"
                # Rust's default hook prints the panic MESSAGE on the line(s)
                # after 'panicked at file:line:col:'. That body carries the
                # actionable detail (e.g. a GC 'invalid type_id ... site=...'
                # diagnostic); the location line alone is not enough to triage
                # a flaky crash from CI logs, so append the first message line.
                for follow in lines[idx + 1 :]:
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

# ── Helpers ──────────────────────────────────────────────────────────

def _jit_stats_merged(stderr):
    """Every `[jit-stats]` line merged into one `key -> value` map, UNFILTERED.

    `None` when the run printed no such line at all, which is a different
    condition from printing one with no keys.

    Split out from `_jit_stats_snapshot` because the two callers want opposite
    things. A committed baseline may only contain the host-stable surface, so
    the snapshot narrows to `JITSTATS_SNAPSHOT_FIELDS`; a *non-vacuity* check
    has to read a counter precisely because it is NOT recorded — see
    `JITSTATS_DENOMINATOR_FIELDS`. Narrowing first would hide it.
    """
    fields = {}
    seen = False
    for line in stderr.splitlines():
        if not line.startswith("[jit-stats]"):
            continue
        seen = True
        for token in line[len("[jit-stats]") :].split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
    return fields if seen else None


def _jit_stats_snapshot(stderr, ungated=()):
    """Return every jit-stats line merged into normalized key/value text.

    The interpreter emits more than one `[jit-stats]` line — the counters, the
    mc_diag tallies, the descr-set universe. Keeping only the last one dropped
    `loops_aborted` and `internal_compile_panics` from the snapshot, which
    silently disarmed `_jit_stats_change`: it read both counters as
    absent on the current side, so `cur > base` could never hold no matter how
    far they rose. Later lines still win on a repeated key.

    Merging is what keeps that true as lines are added. Selecting one line
    instead — the last whose first token is not a bare diagnostic name like
    `mc_diag` / `heap_buckets` / `bridge_diag` / `fbw_diag` / `exec_hist` —
    re-disarms the floor the moment a second key=value line joins the first,
    because the newcomer displaces the counters. The bare-name lines need no
    special case here: their name token carries no `=` and is skipped with
    every other non-assignment token.

    The merged set is then narrowed to `JITSTATS_SNAPSHOT_FIELDS`, because what
    a run may read and what a committed baseline may contain are two different
    questions: merging arms the floor, the filter keeps the recorded surface
    host-stable.

    *ungated* names counters this fixture's header exempted
    (`synth_ungated_jitstats`). They are dropped here rather than at the
    comparison, so the recorded baseline never carries them either: a field
    absent from both sides compares equal, which is what makes one filter serve
    the record and the gate at once.
    """
    fields = _jit_stats_merged(stderr)
    if fields is None:
        return None
    fields = {
        k: v
        for k, v in fields.items()
        if k in JITSTATS_SNAPSHOT_FIELDS and k not in ungated
    }
    # This watches what the JIT compiles, never how well. A regression that
    # changes no structure (for example, an extra spill) is invisible here.
    return "".join(f"{key}={fields[key]}\n" for key in sorted(fields))


def _parse_jit_stats(snapshot):
    """Split a normalized jit-stats snapshot into its `field -> value` map. A
    snapshot that was never taken reads as no fields at all, so every caller can
    treat "absent" and "absent from this side" the same way."""
    if snapshot is None:
        return {}
    return dict(line.split("=", 1) for line in snapshot.splitlines())


# Every counter a `.jitstats` baseline records is gated, in BOTH directions: the
# recorded surface and the gated surface are the same set, so a baseline can
# never carry a number nobody checks. A fixture header may drop a structural
# counter from that set (`synth_ungated_jitstats`), and it leaves both halves at
# once for exactly this reason.
#
# Both directions, because an improvement has to be recorded too. A counter that
# falls is a real change in what the JIT compiles, and leaving the fall ungated
# means the committed baseline quietly stops describing the tree — the next
# regression is then measured against a number that was already stale. Gating a
# fixture on an exact match is upstream's own instrument, not a stricter one
# invented here: `OpMatcher.match` (test_pypy_c/model.py) compares a compiled
# loop operation by operation against a literal expected trace, so an
# improvement that reorders or removes an operation fails that assertion until
# the expected trace is updated. These counters are a coarser projection of the
# same idea.
#
# The exact comparison is affordable, but not because the compile decision is
# free of per-run inputs — it is not. `JitCounter` does index the loop half by
# a hash rather than an address, but the guard half is built from one:
# `current_object_addr_as_int(self) * 777767777 + intval * 1442968193`
# (compile.py:780-781, ported at pyjitpl.rs:10841-10845). Upstream spells it
# the same way, so this is orthodoxy rather than drift, and it is not what
# moved these numbers.
#
# What moved them is that `guard_failures` is not a compile decision at all.
# Each counter is a plain increment at the event (`record_guard_failure_event`,
# pyjitpl.rs), so the total is the runtime taken-count of every guard site —
# the same run reports `mc_diag mc_entered` with the identical value, because
# the counter is the metainterp-entry count. An already-compiled guard running
# once more moves it while `loops_compiled` and `bridges_compiled` stay put,
# which is exactly the shape every disagreement took. The GC schedule was
# reaching it through the back-edge safepoint poll; both halves of that
# schedule are pinned above, and with them pinned the numbers agree across
# hosts and across tree vintages — `recursive_call_frame_relocation` reads 636
# and `closure_per_call` 414 on macOS and on Linux alike, where unpinned the
# same nursery gave 638 on one and 637 on the other.
#
# Measured across runners as well as run-to-run — one CI run reported
# `list_length_hint_validate guard_failures 828 -> 4923` and
# `exception_args_virtual 401 -> 1002` with identical digits on ubuntu-24.04,
# macos-latest and windows-latest. That measurement is what retired the +25%
# band `guard_failures` used to carry for cross-runner drift that had never
# been measured.
#
# `internal_compile_panics` counts an internal compile bug falling back to the
# interpreter, so 0 is its only healthy value. `loops_aborted` is not absolute:
# an abort is either a give-up the tracer is supposed to perform (an over-long
# trace hits `blackhole_if_trace_too_long`, pyjitpl.py) or a walker
# coverage gap that is tracked elsewhere — the synth fixtures below carry
# `LoopBearingCalleeInlineUnsupported`, the walker's open gap on inlining a
# loop-bearing callee. Either way the baseline pins the count those known
# declines produce today, and a rise means a loop the tracer used to finish has
# stopped finishing.
#
# `descr_set_absent`, `descr_set_ambiguous` and `descr_set_stale_absent` are the
# descr-universe invariants. Upstream cannot reach any of the three —
# `effectinfo.py:492-494` builds its sets out of the descr objects `cpu.*descrof`
# just created in the same process, so no member of a raw set can fail to
# resolve — and it expresses this class of condition with a plain `assert`
# (`descr.py:47`, `effectinfo.py:486,525`). Pyre resolves the sets across a
# build-time/runtime split where all three are reachable in principle and the
# runtime answer is a sound degradation rather than a crash, so the assertion is
# expressed as a counter that must not rise off zero. A field absent from a
# baseline reads as 0, so these gate from the first run without re-recording
# anything.
#
# `fbw_rolled_back_with_effects` counts walks that ended UNCOMMITTED after a
# residual had already run an irreversible effect — it wrote live heap outside
# the journals, or it entered a Python frame. The store journal cannot undo
# either, so the legacy replay the caller falls back to applies those effects a
# second time. Upstream has no counterpart state: `_copy_data_from_miframe`
# (blackhole.py) copies the register banks unconditionally and has no failing
# path, so every abort there leaves a resumable image. Pyre's mirror is partial,
# so an abort can arrive with no image to adopt, and the difference is a wrong
# answer rather than a slow one — three such bugs (a locals-typed vable overlay,
# an unpublished escape root stack, an unmodelled LOAD_SPECIAL) were each found
# by reading this counter's population by hand. Gating it is what makes the
# fourth one fail CI instead of surfacing as an intermittent asyncio failure.
# Like `loops_aborted` it is not absolute: the baseline pins the instances that
# exist today and a rise means a new one.
#
# `field_pos_spec_misplaced` and `field_pos_attached_misplaced` are the same kind
# of assertion one layer down: a field descr's `index_in_parent` (`descr.py:228`)
# must name the slot its own parent puts the field in, which upstream gets for
# free because `heaptracker.py:60-72` and `:96-112` are one walker. Both are
# counted where the producer's number is still readable — before
# `get_field_descr` re-derives it — because the counter that looks like it
# already gates this, `field_pos_rederived`, cannot: `derive_index_in_parent`
# overwrites the number it disagrees with, and skips the check entirely whenever
# the parent is not yet published, which is most of them. So a printed
# `field_pos_rederived=0` never meant the producers agreed, and gating it would
# have gated nothing.
JITSTATS_BADNESS_FIELDS = (
    "loops_aborted",
    "internal_compile_panics",
    "descr_set_absent",
    "descr_set_ambiguous",
    "descr_set_stale_absent",
    "fbw_rolled_back_with_effects",
    "field_pos_spec_misplaced",
    "field_pos_attached_misplaced",
    "fbw_store_journal_rollback_failed",
)

# A zero-valued badness counter is meaningful only if its source census ran.
# Pair counters whose census has a reliable denominator and reject runs where
# that denominator is absent or zero. The denominator is not baselined because
# it describes a host-dependent descriptor population; only non-vacuity is
# stable across hosts. `field_pos_attached_misplaced` is intentionally absent:
# its checked population can legitimately be empty for compiled fixtures.
JITSTATS_DENOMINATOR_FIELDS = {
    "field_pos_spec_misplaced": "field_pos_spec_checked",
}

# The count-valued counters, and what a move in either direction means:
#
# * `guard_failures` counts every guard failure that re-enters the metainterp,
#   so it is what moves when a compiled loop's guards start failing more often —
#   a cost `max-pypy-ratio` cannot see, because that ceiling is one absolute
#   number per fixture: recursion_memo_branch sits near 8.9x under a limit of
#   20, so it absorbs a 2x regression and still passes. Narrowing
#   `nested_break_bridge_resume_hazard` so that fixture's `main` compiles moves
#   the counter 1757 -> 2613 (+49%) and measures slower, and no other gate here
#   notices.
# * `loops_compiled` closes the blind spot the others leave open: a change that
#   makes the tracer stop admitting a frame at all — a widened
#   `unsupported_jit_shape` predicate, a pre-trace decline — aborts nothing
#   (`loops_aborted` holds) and *lowers* `guard_failures` (the compiled guards
#   that used to fail are gone), so a fall here is the signal that a hot loop
#   went back to running interpreted.
#   This counts all compilations, including straight-line traces, so it cannot
#   distinguish a loop-shaped trace from a flat trace. Inspect `Label` and
#   `Jump` operations when that distinction matters.
# * `bridges_compiled` is that same signal one level down: guards that stop
#   earning a bridge keep re-entering the metainterp forever. It was left
#   ungated for a long time on the grounds that it "moves in both directions
#   under ordinary tuning", which made a bridge collapse (27 -> 0) invisible for
#   as long as `guard_failures` stayed inside its band — the whole dead-bridge
#   class this suite exists to catch.
# * `retraces_compiled` records that a requested retrace was assembled and
#   attached. A fall means the retrace path stopped producing an artifact;
#   without this positive signal, `loops_aborted` can hold at zero and the
#   accompanying fall in `guard_failures` looks like an improvement.
# * `fbw_blackhole_adopted_single_frame` and
#   `fbw_blackhole_adopted_multi_frame` count successful full-body-walk
#   blackhole adoptions: the walk handed the interpreter a resumable image
#   instead of making its caller replay the region. A fall means the adoption
#   path stopped firing and the legacy replay path came back; a rise means more
#   walks avoided replay and the baseline should say so.
#
# A move is not *always* a defect: two loops merging into one trace lowers
# `loops_compiled` while raising coverage. That is the bargain `loops_aborted`
# already makes — an intentional structural change re-records its baseline, and
# the gate is what forces someone to look at the delta and say why.

# What a committed `.jitstats` baseline is allowed to contain. `_jit_stats_snapshot`
# merges every `[jit-stats]` line so the floor above cannot be disarmed by
# reshuffling them, but the merged set also carries keys that are not comparable
# against a file checked into the repo:
#
# * `all_descrs` and `descr_set_resolved` are host-dependent because Charon/LLBC
#   extraction runs per host. Committing either count would make otherwise
#   equivalent hosts disagree.
# * the `mc_diag` tallies include contention counters (`busy_skip`,
#   `stfe_cell_busy`, `stfe_tick`) that move with machine load.
# * `PYRE_WASM_JIT_STATS=1` adds wall-clock (`compile_ms`) and repeated
#   `exec_hist` lines whose keys collide across lines; check.py never sets it,
#   but a developer who has it exported would otherwise poison a `--snapshot`
#   re-record.
#
# The badness fields are all zero in a healthy run and the three structural
# counts are host-stable, so this surface needs no re-record when a new
# diagnostic line is added.
JITSTATS_SNAPSHOT_FIELDS = JITSTATS_BADNESS_FIELDS + (
    "loops_compiled",
    "bridges_compiled",
    "retraces_compiled",
    "guard_failures",
    "fbw_blackhole_adopted_single_frame",
    "fbw_blackhole_adopted_multi_frame",
)

# `back_edge_polls` is deliberately absent, and is the one counter that must
# stay absent. It reports how many times a compiled loop left machine code
# because the eval-breaker word was armed — a measure of when a collection
# landed, not of anything the compiler decided. Recording it would move the
# schedule-sensitivity this split was made to remove onto a new key instead of
# removing it. It is printed on the `[jit-stats]` line either way, so a reader
# diagnosing a `guard_failures` move can still see it.


# Which way each counter has to move to be a regression rather than a gain.
# Both outcomes fail — a baseline that stopped describing the tree has to be
# re-recorded either way, and letting a gain pass is how the eight baselines
# this split was written for went stale. The labels exist so a reader can tell
# the two apart at a glance: REGRESSED is "something got worse, go look",
# IMPROVED is "something got better, re-record it".
#
# `loops_compiled` is inverted against the badness fields: it is the counter
# that falls when the tracer stops admitting a frame at all, which aborts
# nothing and *lowers* `guard_failures`, so a fall is the regression and a rise
# is the gain — within the limit recorded above, that it counts compiles and not
# loop-shaped ones. `retraces_compiled` has the same polarity: a fall means an
# assembled retrace stopped being attached. The blackhole adoption counters are
# inverted the same way: a fall means the interpreter stopped receiving an
# image and went back to replay.
JITSTATS_REGRESSION_ON_RISE = JITSTATS_BADNESS_FIELDS + ("guard_failures",)
JITSTATS_REGRESSION_ON_FALL = (
    "loops_compiled",
    "retraces_compiled",
    "fbw_blackhole_adopted_single_frame",
    "fbw_blackhole_adopted_multi_frame",
)

# How many fresh runs a disagreeing fixture gets before its counters are called
# stable. Only a fixture that already disagrees pays this, so the common path is
# untouched.
#
# The gate's premise is that these counters are a function of the program. Where
# that holds, a repeat reproduces the first run exactly and the disagreement is
# gated as before. Where it does not — measured on this tree: two runs of the
# same `pyre-dynasm` binary, minutes apart on a loaded machine, disagreed by
# `loops_compiled` +2 on five unrelated fixtures — the number being compared is
# not a property of the tree, and failing on it reports load as if it were a
# code change.
#
# So instability is MEASURED, never declared: nothing is annotated as flaky, and
# a fixture stops being gated only by demonstrating, in this same invocation,
# that it does not reproduce itself. The cost is that a genuinely
# nondeterministic regression now warns instead of failing — which is the right
# trade only because a counter that will not reproduce cannot be gated on
# exactly anyway, and the warning still names it.
JITSTATS_STABILITY_RUNS = 2

# `bridges_compiled` is deliberately in neither: it is not interpretable on its
# own, in either direction. A fall to 0 is the dead-bridge regression this gate
# was built to catch when guards still fail, and a gain when the guards stopped
# failing (`list_length_hint_validate` fell 4 -> 0 as guard_failures fell
# 828 -> 1). A rise is more guards crossing the bridge threshold, which is
# either wider coverage or a guard storm. Undecidable counters are reported as
# regressions, so the ambiguous case is the one a human is asked to look at.


def _jit_stats_change(saved, current, bands=None):
    """Return `(regressions, improvements, banded)` change descriptions.

    Both directions gate: a rise means the JIT started aborting traces, hitting
    internal compile panics or failing guards it did not before, and a fall
    means it stopped compiling something it used to compile. Neither may pass
    unrecorded — see the surface comment above.

    A band reclassifies an integer-valued move at or below its symmetric width,
    whether that move would be a regression or an improvement. The move is
    reported in `banded` rather than hidden. A band cannot reclassify a
    non-integer value or a move outside the band, and the fixture parser rejects
    bands on badness counters whose healthy value must remain exactly zero.

    A field missing from either side reads as "0", so a baseline recorded before
    a counter existed still matches a run that reports it as 0, and adding an
    invariant counter costs no re-record. The cost of that convenience is that a
    field absent from BOTH sides compares equal forever: a backend whose
    [jit-stats] line stops naming a counter disarms that counter's gate on every
    one of its baselines, silently. Whenever a backend's line changes shape,
    re-record its whole baseline surface rather than trusting the run that
    follows."""
    bands = bands or {}
    old_fields = _parse_jit_stats(saved)
    new_fields = _parse_jit_stats(current)
    regressions, improvements, banded = [], [], []
    for field in JITSTATS_SNAPSHOT_FIELDS:
        old, new = old_fields.get(field, "0"), new_fields.get(field, "0")
        if old == new:
            continue
        try:
            old_int, new_int = int(old), int(new)
        except ValueError:
            # A counter that stopped being an integer is not a gain.
            rose = None
        else:
            if field in bands and abs(new_int - old_int) <= bands[field]:
                banded.append(
                    f"{field} {old} -> {new} (within band {bands[field]})"
                )
                continue
            rose = new_int > old_int
        if rose is not None and (
            (rose and field in JITSTATS_REGRESSION_ON_FALL)
            or (not rose and field in JITSTATS_REGRESSION_ON_RISE)
        ):
            improvements.append(f"{field} {old} -> {new}")
        else:
            regressions.append(f"{field} {old} -> {new}")
    return regressions, improvements, banded


def _jit_stats_vacuous(stderr):
    """Return why a gated badness counter is vacuous this run, or "".

    A badness counter's zero means "clean" only when its source census ran.
    Require a nonzero denominator without baselining its host-dependent value.
    This detects total loss of a census, not a producer that merely omits some
    census updates.
    """
    fields = _jit_stats_merged(stderr)
    if fields is None:
        return ""  # the caller already failed this run for the missing line
    for numerator, denominator in JITSTATS_DENOMINATOR_FIELDS.items():
        raw = fields.get(denominator)
        if raw is not None and raw.isdigit() and int(raw) > 0:
            continue
        return (
            f"{denominator}={raw if raw is not None else '<absent>'} — "
            f"the census behind `{numerator}` did not run, so that counter's 0 "
            f"means 'nothing was checked', not 'nothing is misplaced', and its "
            f"gate passed without inspecting anything"
        )
    return ""


def _jit_stats_context(current):
    """Return " (observed loops_compiled=N bridges_compiled=M)", or "".

    Report what the run compiled alongside what moved. A `guard_failures` change
    costs nothing to explain when the same run also compiled more loops or
    bridges — each new guard fails `trace_eagerness` times before its bridge
    attaches — but a gate that names only the fields that moved leaves the
    reader unable to tell that case from a real defect without re-running the
    bench. On a platform the developer does not have (the CI runners),
    re-running is not an option and the row is unjudgeable without this.
    """
    fields = _parse_jit_stats(current)
    context = " ".join(
        f"{f}={fields[f]}"
        for f in ("loops_compiled", "bridges_compiled")
        if f in fields
    )
    return f" (observed {context})" if context else ""


def scaled_timeout(base, scale):
    v = base * scale
    return int(v) if v == int(v) else float(f"{v:.3f}".rstrip("0").rstrip("."))


def fmt_time(t):
    if t is None or t == "-":
        return "-"
    if isinstance(t, (int, float)):
        return f"{t:.2f}s"
    try:
        return f"{float(t):.2f}s"
    except ValueError:
        return t


def synth_perf_gate(path):
    """Read an optional per-fixture performance limit from its header.

    Synthetic tests own their limits, for example:
        # pyre-check: max-pypy-ratio=30
    Keeping the gate beside the workload makes a changed loop count or known
    slow path reviewable with the test that needs the allowance.

    The limit is read against the native backends only; `run_synthetic_bench`
    exempts wasm.

    A fixture states its ceiling and nothing else: the floor that goes with it
    is `perf_gate_floor`'s business, so a leftover `min-pypy-ratio` header is
    rejected rather than ignored.
    """
    prefix = "# pyre-check: max-pypy-ratio="
    retired = "# pyre-check: min-pypy-ratio="
    ratio = None
    with open(path, encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if line.startswith(retired):
                raise ValueError(
                    f"{path} states a pypy floor: {line.strip()}\n"
                    "the floor is derived from max-pypy-ratio; delete the line"
                )
            if not line.startswith(prefix) or ratio is not None:
                continue
            try:
                ratio = float(line[len(prefix):].strip())
            except ValueError as e:
                raise ValueError(f"invalid synthetic performance gate in {path}: {line.strip()}") from e
            if ratio <= 0:
                raise ValueError(f"synthetic performance gate must be positive in {path}")
    return ratio


def wasm_ratio_gate(path):
    """Read an optional per-fixture ceiling on wasm's ratio to dynasm:
        # pyre-check: max-wasm-ratio=6

    Absence means `WASM_MAX_DYNASM_RATIO`, not "no ceiling" -- the opposite of
    `synth_perf_gate` and `synth_rss_gate`, whose absence exempts a fixture
    entirely. A directive here is therefore an allowance carved out of a gate
    that already applies, so a run names every fixture that used one: an
    allowance that is quietly the reason a suite is green is worse than no gate.

    Read for every bench rather than synthetic ones alone, because the fixtures
    that need an allowance are not all synthetic.

    What the fixtures carrying one have in common: the wasm backend reaches
    CPython-level objects through interpreter round-trips and allocates each on
    a Rust heap the wasm path does not yet collect, so a loop dominated by that
    work runs several times dynasm's execution time for a reason that is
    structural rather than a regression. The allowances are set at the highest
    ratio observed plus 15%, and each fixture records the ratios it was fitted
    to. ubuntu-24.04 is the only runner that builds wasm, so it supplies most of
    them; a fixture sitting on the gate locally is fitted here instead.

    15% rather than the ~3% ubuntu shows between runs because the ratio moves
    with host load even though both sides are user-CPU and measured in the same
    invocation: fannkuch reads 2.9x on an idle machine here and 3.1x on the same
    machine under a load average of 41. An allowance fitted to an idle reading
    would be a gate that fails on a busy runner.
    """
    prefix = "# pyre-check: max-wasm-ratio="
    ratio = None
    with open(path, encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if not line.startswith(prefix) or ratio is not None:
                continue
            try:
                ratio = float(line[len(prefix):].strip())
            except ValueError as e:
                raise ValueError(f"invalid wasm ratio gate in {path}: {line.strip()}") from e
            if ratio <= 0:
                raise ValueError(f"wasm ratio gate must be positive in {path}")
    return ratio


def perf_gate_floor(ceiling):
    """The pypy ratio a bench with this ceiling must not read below.

    See `PERF_GATE_FLOOR_RATIO`: a bench is asked to reach parity, never to
    stay slow, so the floor is parity itself until the ceiling comes close
    enough that parity would sit inside the allowance.
    """
    return min(PERF_GATE_FLOOR_RATIO, ceiling / PERF_GATE_FLOOR_DIVISOR)


def synth_rss_gate(path):
    """Read an optional per-fixture peak-RSS ceiling from its header:
        # pyre-check: max-rss-mb=200

    A second axis beside `max-pypy-ratio`, not a replacement: the ratio gate
    is blind to memory, so a change that trades retained bytes for speed
    passes it while regressing badly. Frames are the standing example —
    every interpreted call retains an old-gen frame, and nothing in this
    harness has ever measured that.

    Absent directive means no ceiling, so adding this gate cannot fail a
    fixture that does not opt in. Read against native backends only, like
    the ratio gate.
    """
    prefix = "# pyre-check: max-rss-mb="
    with open(path, encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if not line.startswith(prefix):
                continue
            try:
                limit = float(line[len(prefix):].strip())
            except ValueError as e:
                raise ValueError(f"invalid synthetic memory gate in {path}: {line.strip()}") from e
            if limit <= 0:
                raise ValueError(f"synthetic memory gate must be positive in {path}")
            return limit
    return None


def synth_skip_backends(path):
    """Read an optional per-fixture backend exemption from its header:
        # pyre-check: skip-backends=wasm

    For a fixture that guards a mechanism a backend does not implement at
    all, so the fixture can only ever restate that known gap. The named
    backends print `skip` instead of running, the way
    `run_bench(skip_backends=...)` already does.

    This is not a way to park a failure: the header line must be followed by
    a comment saying which mechanism is missing, so the exemption is
    reviewable next to the workload. Unknown backend names are an error, not
    a silent no-op — a typo would otherwise read as "exempted". A directive
    below the 20-line window is likewise not silently honored: the fixture
    simply runs everywhere and goes red, which is the safe direction.

    The exemption covers a backend's *execution* only. It deliberately does
    NOT cover the three baseline-failure paths below (cpython crash, pypy
    crash, cpython/pypy output mismatch): those mean the fixture itself is
    broken, and that must stay loud on every backend rather than be hidden
    behind an exemption written for a different reason.
    """
    prefix = "# pyre-check: skip-backends="
    with open(path, encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if not line.startswith(prefix):
                continue
            names = [n.strip() for n in line[len(prefix):].split(",")]
            names = [n for n in names if n]
            if not names:
                raise ValueError(f"empty synthetic backend exemption in {path}: {line.strip()}")
            unknown = [n for n in names if n not in ALL_BACKENDS]
            if unknown:
                raise ValueError(
                    f"unknown backend(s) {unknown} in synthetic backend exemption in {path}: "
                    f"{line.strip()}"
                )
            return tuple(names)
    return ()


def synth_jitstats_bands(path):
    """Read an optional per-fixture jit-stats band from its header:
        # pyre-check: jitstats-band=guard_failures=8

    The band is symmetric around the recorded baseline and absorbs moves in
    both directions: a delta within the band is not signal either way. An
    absorbed move is reported on that run rather than swallowed, so the
    allowance is visible whenever it is used and not only in the fixture
    header. It is only for schedule-sensitive counters; badness counters must
    remain exactly zero. The directive must be followed by a comment describing
    the measured variance, so the allowance is reviewable next to the workload.
    Unknown or duplicate fields, repeated directives, and non-positive or
    non-integer widths are errors rather than silent no-ops. A directive below
    the 20-line window is simply not read, so the fixture gates normally in
    that case.
    """
    prefix = "# pyre-check: jitstats-band="
    bands = None
    with open(path, encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if not line.startswith(prefix):
                continue
            if bands is not None:
                raise ValueError(f"duplicate jit-stats band directive in {path}: {line.strip()}")
            bands = {}
            entries = line[len(prefix):].strip().split(",")
            for entry in entries:
                parts = entry.split("=")
                if len(parts) != 2 or not all(part.strip() for part in parts):
                    raise ValueError(f"invalid jit-stats band in {path}: {line.strip()}")
                field, raw_width = (part.strip() for part in parts)
                if field not in JITSTATS_SNAPSHOT_FIELDS:
                    raise ValueError(
                        f"unknown jit-stats band field {field!r} in {path}: {line.strip()}"
                    )
                if field in JITSTATS_BADNESS_FIELDS:
                    raise ValueError(
                        f"badness counter cannot be banded in {path}: {line.strip()}"
                    )
                if field in bands:
                    raise ValueError(
                        f"duplicate jit-stats band field {field!r} in {path}: {line.strip()}"
                    )
                try:
                    width = int(raw_width)
                except ValueError as e:
                    raise ValueError(
                        f"invalid jit-stats band width in {path}: {line.strip()}"
                    ) from e
                if width <= 0:
                    raise ValueError(
                        f"jit-stats band width must be positive in {path}: {line.strip()}"
                    )
                bands[field] = width
    return bands if bands is not None else {}


def _synth_header_flag(path, directive, malformed):
    """Read a valueless per-fixture header flag from the first 20 lines.

    Shared by the two cpython opt-outs below, which differ only in the
    directive they spell and what a malformed spelling is called. Not honored
    below the 20-line window, which is the safe direction: the fixture simply
    runs cpython too.
    """
    with open(path, encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if line.strip() == directive:
                return True
            if line.startswith(directive):
                raise ValueError(f"{malformed} in {path}: {line.strip()}")
    return False


def synth_skip_cpython(path):
    """Read an optional per-fixture cpython opt-out from its header:
        # pyre-check: skip-cpython

    For a fixture sized past what cpython can usefully run. The ratio gate is
    pypy's; cpython is only a reference, and once the workload is big enough
    for pypy's execution time to be a measurement rather than a clock tick,
    cpython can be minutes behind. `SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S`
    already drops it in that case, but only after spending that timeout on
    every run — this says so up front.

    What it gives up is the cpython/pypy output cross-check, which is the one
    thing that catches a fixture whose two baselines disagree. So the header
    line must be followed by a comment saying why the size is what it is,
    the way `skip-backends` does, and a fixture cpython can still run has no
    business carrying it.

    Not honored below the 20-line window, which is the safe direction: the
    fixture simply runs cpython too.
    """
    return _synth_header_flag(
        path,
        "# pyre-check: skip-cpython",
        "synthetic cpython opt-out takes no value",
    )


def synth_no_cpython(path):
    """Read an optional per-fixture exemption from the reference oracle:
        # pyre-check: no-cpython

    A synthetic fixture is normally rejected outright when the reference and
    pypy disagree, because a workload the two do not agree on cannot say
    anything about pyre. That leaves a DELIBERATE divergence from the
    reference untestable anywhere in the tree: a parity fixture has to pass
    under the reference as well, and the vendored suite measures a divergence
    as a regression rather than as coverage. So behaviour pyre takes from pypy
    precisely because the reference does something else — `type.__basicsize__`
    and its three siblings being absent, `__flags__` carrying pypy's six-bit
    mask, `__sizeof__` not existing — had nowhere to be pinned.

    This header names pypy as the fixture's only oracle: the reference is not
    run, and the output is compared against pypy alone. It is only correct for
    a fixture whose whole subject IS such a divergence; anything else wants
    both references, and the disagreement it would otherwise report is the
    point of that check. Follow the header with a comment naming the
    divergence, so the exemption is reviewable next to the workload.
    """
    return _synth_header_flag(
        path,
        "# pyre-check: no-cpython",
        "malformed reference-oracle exemption",
    )


def synth_selfcheck(path):
    """Read an optional self-checking fixture marker from its header:
        # pyre-check: selfcheck

    These fixtures assert their own invariant and print ``PASS`` instead of
    producing stable output for the cpython/pypy parity comparison.  Keeping
    the marker on the fixture lets the synthetic suite discover it normally
    without a second, hard-coded list in ``main``.
    """
    return _synth_header_flag(
        path,
        "# pyre-check: selfcheck",
        "synthetic selfcheck marker takes no value",
    )


def synth_ungated_jitstats(path):
    """Read an optional per-fixture jit-stats exemption from its header:
        # pyre-check: ungated-jitstats=bridges_compiled,guard_failures

    For a fixture whose named counters are demonstrably not a function of the
    tree — the same binary, the same fixture and the same child environment
    read two different values. The names are dropped from both sides of the
    comparison and from the recorded baseline, so the file carries no number
    nobody checks, which is the rule the recorded surface is built on.

    Only the structural counters may be named. The badness fields are
    assertions that must hold in every run (`internal_compile_panics` is an
    internal compile bug, the descr-universe counters are invariants upstream
    spells as `assert`), and a run that reaches one has a defect no
    per-run input excuses, so naming one here is an error rather than an
    exemption. An unknown name is an error too — a typo would otherwise read
    as "exempted".

    Like the other directives, one below the 20-line window is not honored:
    the fixture simply stays gated, which is the safe direction. The header
    must also carry what was measured, so the next reader can check the claim
    rather than take it.

    Read from `_apply_snapshot_gate`, so unlike the rest of this family it
    reaches the regular benches too — the gate it exempts is the one they
    share, and a counter that will not reproduce is not a synthetic-only
    condition.
    """
    prefix = "# pyre-check: ungated-jitstats="
    with open(path, encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if not line.startswith(prefix):
                continue
            names = [n.strip() for n in line[len(prefix):].split(",")]
            names = [n for n in names if n]
            if not names:
                raise ValueError(f"empty jit-stats exemption in {path}: {line.strip()}")
            unknown = [n for n in names if n not in JITSTATS_SNAPSHOT_FIELDS]
            if unknown:
                raise ValueError(
                    f"unknown counter(s) {unknown} in jit-stats exemption in {path}: "
                    f"{line.strip()}"
                )
            badness = [n for n in names if n in JITSTATS_BADNESS_FIELDS]
            if badness:
                raise ValueError(
                    f"badness counter(s) {badness} cannot be exempted in {path}: "
                    f"{line.strip()}"
                )
            return tuple(names)
    return ()


def default_binary(backend):
    name = CARGO_CONFIG[backend]["bin"]
    return f"./target/release/{name}{EXE}"


# Relative tolerance for wasm float outputs ONLY (see `wasm_outputs_match`).
WASM_FLOAT_RTOL = 1e-9


def wasm_outputs_match(output, expected):
    """Bench-output comparison for the wasm backend, allowing a bounded
    float divergence — used ONLY for benches explicitly opted in with
    run_bench(..., wasm_float_tol=True). No blanket wasm allowance exists;
    today only nbody is marked, and every other wasm bench stays byte-exact.

    Root cause (measured, not guessed): the wasm guest and native pyre run
    the SAME libm source, but libm's transcendentals (`pow` is the FreeBSD
    `e_pow.c` hi/lo split) bottom out on ARCH-SPECIFIC building blocks
    (`libm/src/math/arch/aarch64.rs`). On aarch64 those land on hardware-tuned
    ops that match the platform macOS libm — so native pyre matches
    CPython/PyPy bit-for-bit — while wasm32 gets the generic software
    fallbacks. That is a ~0.5-ULP-per-op gap that accumulates (nbody: 5M pow
    calls -> ~1490 ULP, 3e-13 relative, in the printed energy). It is an
    unfixable target-ISA codegen gap, not a miscompile: an FMA-fusion
    hypothesis was tested and refuted (fp-contract=fast on the wasm build is
    a no-op — wasm32 scalar has no FMA instruction — and libm's pow uses no
    mul_add), forcing native unfused would break native parity with the
    platform reference, and host-libm callbacks were rejected (they make wasm
    output vary by host machine, which we do not want).

    So float tokens are compared with a relative tolerance `WASM_FLOAT_RTOL`
    (1e-9 — ~4 orders looser than the observed 3e-13 drift, ~3 orders
    TIGHTER than the smallest real value bug seen, 5.7e-6). Every non-float
    token (ints, strings, non-finite floats) still must match byte-for-byte,
    so an int off-by-one can never slip through."""
    if output == expected:
        return True
    out_lines, exp_lines = output.splitlines(), expected.splitlines()
    if len(out_lines) != len(exp_lines):
        return False
    for out_line, exp_line in zip(out_lines, exp_lines):
        if out_line == exp_line:
            continue
        out_toks, exp_toks = out_line.split(), exp_line.split()
        if len(out_toks) != len(exp_toks):
            return False
        for out_tok, exp_tok in zip(out_toks, exp_toks):
            if out_tok == exp_tok:
                continue
            # Only finite float-shaped tokens (decimal point or exponent)
            # get tolerance; ints/strings/nan/inf must match byte-for-byte
            # above, so an int off-by-one can never slip through.
            def _floaty(tok):
                return "." in tok or "e" in tok or "E" in tok
            if not (_floaty(out_tok) and _floaty(exp_tok)):
                return False
            try:
                a, b = float(out_tok), float(exp_tok)
            except ValueError:
                return False
            if not (math.isfinite(a) and math.isfinite(b)):
                return False
            if abs(a - b) > WASM_FLOAT_RTOL * max(abs(a), abs(b)):
                return False
    return True


# Backends rendered in fixed-column displays, in order. Any enabled backend not
# listed here still runs and is counted; it just falls outside the fixed columns.
ALL_BACKENDS = ("dynasm", "cranelift", "wasm")


def _wasm_target_installed():
    """Whether the wasm backend can be built here.

    The only extra prerequisite over the native backends is the
    `wasm32-unknown-unknown` rustup target (the wasmtime runtime is embedded
    in `pyre-wasm-runner`, not an external tool). If it is missing, the wasm
    build would `rustup target add`-fail, so wasm stays out of the default set.
    """
    try:
        proc = subprocess.run(
            ["rustup", "target", "list", "--installed"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and "wasm32-unknown-unknown" in proc.stdout.split()


# wasm joins the defaults only where its target is installed, so a plain
# `check.py` on an unconfigured machine still runs just the native backends.
DEFAULT_BACKENDS = ("dynasm", "cranelift")
if _wasm_target_installed():
    DEFAULT_BACKENDS = (*DEFAULT_BACKENDS, "wasm")

# ── Check runner ─────────────────────────────────────────────────────

class Check:
    def __init__(self, args):
        self.args = args
        self.results = []
        self.comparisons = []
        # Synthetic fixtures whose cpython reference run exceeded its budget,
        # so pypy alone was the baseline for them.
        self.cpython_reference_skips = []
        self.cpython_declared_skips = []
        # Per-backend bookkeeping, keyed by backend name so any backend
        # (dynasm / cranelift / wasm / a single `--pyre-path` binary) is tracked
        # uniformly.
        self.pass_count = {}
        self.fail_count = {}
        self.pyre = {}
        # interpreter/backend key -> measured empty-program user-CPU startup (s).
        # Populated by measure_startups(); missing key => 0.0 (no subtraction).
        self.startup = {}
        # (backend, bench name) -> raw user-CPU of that backend's run in THIS
        # invocation. Feeds the wasm-vs-dynasm gate, which needs a baseline
        # measured on the same host under the same load; a recorded one would
        # age out of the machine it was taken on.
        self.bench_elapsed = {}
        # Fixtures where the wasm run had no dynasm time to divide by -- absent
        # or under the noise floor -- so WASM_MAX_DYNASM_RATIO was never
        # applied. Reported in the summary: an unevaluated gate otherwise
        # prints the same green as a satisfied one.
        self.wasm_ratio_ungated = []
        # (bench name, ceiling) for fixtures whose header raised the ratio
        # above WASM_MAX_DYNASM_RATIO, for the same reason as the line above: a
        # gate widened for a fixture and a gate the fixture satisfied both print
        # green. Reported from `print_summary` rather than beside that line,
        # because three of the fixtures carrying an allowance are regular
        # benches and a `--no-synthetic` run would otherwise not name them.
        self.wasm_ratio_allowed = []
        self.snapshot_diffs = []
        self.snapshot_missing = []
        self.jitstats_diffs = []
        # Benches whose gated counters all moved the way their field calls a
        # gain. Still failures — the baseline stopped describing the tree — but
        # tracked apart so the summary can say which reds need investigating and
        # which only need re-recording.
        self.jitstats_improvements = []
        # Benches whose counters did not reproduce across repeats in this same
        # invocation. Reported, never failed — see JITSTATS_STABILITY_RUNS.
        self.jitstats_unstable = []
        # Benches whose gated counters moved only inside a declared per-fixture
        # band. Reported, never failed.
        self.jitstats_banded = []
        # "<backend>/<bench>: <counters>" for fixtures whose header exempted a
        # counter (`synth_ungated_jitstats`). An exempted counter and a matching
        # one print the same green, so the exemptions are named in the summary:
        # a gate nobody applied should not read as a gate nobody had to.
        self.jitstats_ungated = []
        self.jitstats_missing = []
        # Benches whose run printed no `[jit-stats]` line at all. Tracked apart
        # from `jitstats_missing` because the two say opposite things: a missing
        # baseline is a bench nobody recorded, an absent line is a bench that
        # stopped reporting.
        self.jitstats_absent = []
        # Benches whose run printed the line, but with a badness counter's
        # census denominator at zero — the counter's gate passed on an
        # instrument that inspected nothing. Tracked apart from
        # `jitstats_absent` because the line IS there and every other gate on it
        # is still meaningful; only the paired counters are void.
        self.jitstats_vacuous = []

    # ── backend helpers ──

    def enabled(self, backend):
        return bool(self._pyre(backend))

    def _pyre(self, backend):
        return self.pyre.get(backend, "")

    def _timeout_scale(self, backend):
        if backend == "dynasm" and self.args.dynasm_timeout_scale is not None:
            return self.args.dynasm_timeout_scale
        if backend == "cranelift" and self.args.cranelift_timeout_scale is not None:
            return self.args.cranelift_timeout_scale
        if backend == "wasm":
            if self.args.wasm_timeout_scale is not None:
                return self.args.wasm_timeout_scale
            # Default wasm headroom composes with --timeout-scale.
            return WASM_TIMEOUT_SCALE * self.args.timeout_scale
        if sys.platform == "win32":
            return WIN_NATIVE_TIMEOUT_SCALE * self.args.timeout_scale
        return self.args.timeout_scale

    def measure_startups(self):
        """Measure each timed interpreter/backend's empty-program user-CPU cost.

        Runs an empty script STARTUP_SAMPLES times per interpreter and records
        the median user-CPU in self.startup. This is the fixed per-process cost
        (interpreter init; for wasm, wasmtime recompiling the module) added to
        every bench regardless of workload; subtracting it yields an
        execution-only comparison. No-op under --no-startup-subtract.
        """
        if self.args.no_startup_subtract:
            return
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False
        ) as handle:
            empty_path = handle.name  # zero-length program
        try:
            targets = [
                ("cpython", [PYTHON3, empty_path], None),
                ("pypy", [PYPY3, empty_path], None),
            ]
            for backend in ALL_BACKENDS:
                if self.enabled(backend):
                    targets.append(
                        (backend, [self._pyre(backend), empty_path], pyre_env())
                    )
            parts = []
            for key, cmd, env in targets:
                samples = []
                for _ in range(STARTUP_SAMPLES):
                    _out, elapsed, code, _err = run_timed(
                        cmd, timeout_s=60, env=env,
                    )
                    if code != 0:
                        samples = []
                        break
                    samples.append(elapsed)
                startup = statistics.median(samples) if samples else 0.0
                self.startup[key] = startup
                parts.append(f"{key}={startup:.3f}s")
            print(dim(
                f"startup (empty-program user-CPU, median {STARTUP_SAMPLES}; "
                f"ratios/gates are execution-only): " + " ".join(parts)
            ))
        finally:
            try:
                os.unlink(empty_path)
            except OSError:
                pass

    def _exec_time(self, key, t):
        """Startup-subtracted user-CPU for interpreter *key*, floored.

        Returns *t* unchanged when subtraction is disabled or *t* is
        unavailable ("-"/None). Never returns <= 0 (floored to
        EXEC_TIME_FLOOR_S) so downstream ratios cannot divide by ~0.
        """
        if t is None or t == "-":
            return t
        value = float(t)
        if self.args.no_startup_subtract:
            return value
        return max(value - self.startup.get(key, 0.0), EXEC_TIME_FLOOR_S)

    def _set_pyre(self, backend, path):
        self.pyre[backend] = path

    # ── comparison table ──

    def _comp_index(self, name):
        for i, c in enumerate(self.comparisons):
            if c["name"] == name:
                return i
        return -1

    def _append_comparison(self, backend, name, t_cpython, t_pypy, pyre_field, note=""):
        idx = self._comp_index(name)
        if idx == -1:
            entry = {
                "name": name,
                "cpython": fmt_time(t_cpython),
                "pypy": fmt_time(t_pypy),
            }
            for b in ALL_BACKENDS:
                entry[b] = "-"
            self.comparisons.append(entry)
            idx = len(self.comparisons) - 1
        else:
            self.comparisons[idx]["cpython"] = fmt_time(t_cpython)
            self.comparisons[idx]["pypy"] = fmt_time(t_pypy)

        cell = pyre_field
        if note:
            note = note.strip("()")
            if note.endswith(" vs pypy"):
                note = note[: -len(" vs pypy")]
            cell = f"{pyre_field:>6s}   {note:>5s}"
        self.comparisons[idx][backend] = cell

    # ── record result ──

    def _record(self, backend, passed, name, detail):
        if passed:
            self.pass_count[backend] = self.pass_count.get(backend, 0) + 1
        else:
            self.results.append(f"{red('FAIL')} {backend} {name}  {detail}")
            self.fail_count[backend] = self.fail_count.get(backend, 0) + 1

    # ── snapshot gate ──

    def _snapshot_path(self, backend, name, suffix):
        return Path(SNAP_DIR) / backend / f"{name}.{suffix}"

    def _jitstats_repeats(self, backend, script, timeout, ungated=()):
        """Re-run the fixture and return each repeat's jit-stats snapshot.

        Returns None if a repeat did not exit cleanly, so a crashing re-run
        cannot argue a real diff away — the caller then gates on the first run
        as if no repeat had happened.

        *ungated* is the caller's, so a repeat is narrowed the same way the run
        being compared was: narrowing one side only would report every exempted
        counter as instability.
        """
        effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))
        snapshots = []
        for _ in range(JITSTATS_STABILITY_RUNS):
            _output, _elapsed, code, stderr = run_timed(
                [self._pyre(backend), script],
                timeout_s=effective_timeout, env=pyre_env(),
            )
            if code != 0:
                return None
            snapshots.append(_jit_stats_snapshot(stderr, ungated))
        return snapshots

    def _jitstats_baseline_path(self, backend, script):
        # The committed structural-stats baseline sits beside its benchmark
        # source (pyre/bench/<name>.<backend>.jitstats), not in the gitignored
        # check.snap scratch tree that holds the local .out/.time snapshots.
        #
        # A `<name>.<backend>.<platform>.github-actions.jitstats` beside it wins
        # on that GitHub runner; otherwise a
        # `<name>.<backend>.<platform>.jitstats` wins on the platform. These are
        # for counters a host genuinely disagrees on, and keep the comparison
        # exact everywhere instead of widening the gate for everyone: a band
        # that tolerated the disagreement would tolerate it on every fixture
        # and all three backends, and the band this gate replaced is exactly
        # what let eight baselines go stale unnoticed.
        #
        # No fixture has one at present, and the last set to need them is worth
        # recording because of how it ended. `str_fstring` was carried by three:
        # the runners read guard_failures 658/659 on ubuntu dynasm/cranelift,
        # the inverse 659/658 on macos, and 658/658 on windows, with every other
        # gated counter identical (six loops, three bridges) and each mismatch
        # reproducing in both stability reruns. The split was stable per host, so
        # it read as a host disagreement.
        #
        # It was not one. The knife edge is a major-collection crossing landing
        # either side of the peeled preamble (see `pyre_env`), and its input is
        # how much the process allocated before the loops got hot — a quantity
        # every host computes differently and nothing about the platform fixes.
        # Adding check.py's two wasm environment keys, semantically ignored by
        # the native backend, moves a local macOS cranelift build 658 -> 659
        # while the number of major collections stays at 11; the guard sequence
        # gains one trace-6 fail-5 exit that the compiled-trace log maps to the
        # GuardFalse on the eval-breaker poll. Raising `PYPY_GC_MIN` past the
        # fixture's working set removed the crossing, and all three overlays
        # came out with it.
        #
        # That is the shape to look for before writing one: a counter that
        # splits per host and moves under a perturbation with no semantic
        # content is not a host disagreement, it is a boundary the harness has
        # not pinned yet. An overlay would have frozen it per platform instead.
        #
        # `child_env_base` since removed the environment term of that volume:
        # a child starts from an allowlist rather than a copy of the runner's
        # environment. That closes one input, not the class -- any other change
        # in allocation volume still moves the point, which is why taking the
        # fixture off the threshold is what actually settled it.
        #
        # An overlay also shadows the shared baseline permanently, so one written
        # against a value that later converges becomes a failure main does not
        # have: `recursive_call_frame_relocation` was given one for 637, and
        # windows later converged on its shared value. Read all three runners
        # back before adding one, and prefer removing the fixture's dependence on
        # the host input to recording the host.
        source = Path(script)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            github_runner = source.with_name(
                f"{source.stem}.{backend}.{sys.platform}.github-actions.jitstats"
            )
            if github_runner.exists():
                return github_runner
        per_platform = source.with_name(f"{source.stem}.{backend}.{sys.platform}.jitstats")
        if per_platform.exists():
            return per_platform
        return source.with_name(f"{source.stem}.{backend}.jitstats")

    def _apply_snapshot_gate(
        self, backend, name, script, output, stderr, elapsed, timeout,
    ):
        status, reason = "ok", ""
        out_path = self._snapshot_path(backend, name, "out")
        time_path = self._snapshot_path(backend, name, "time")
        jitstats_path = self._jitstats_baseline_path(backend, script)
        ungated = synth_ungated_jitstats(script)
        if ungated:
            self.jitstats_ungated.append(f"{backend}/{name}: {', '.join(ungated)}")
        jitstats = _jit_stats_snapshot(stderr, ungated)
        jitstats_bands = synth_jitstats_bands(script)

        # The jit-stats gate — enforced on EVERY run, so a structural JIT change
        # reddens the default `pyre/check.py` (locally, and in the bare CI
        # invocation) the instant it lands, with no --snapshot-diff needed. This
        # catches the inline-abort class (e.g. an inlined-callee LOAD_GLOBAL fold
        # that stops resolving: loops_aborted 0 -> 5) and the dead-bridge class
        # (bridges_compiled 27 -> 0). Record mode is writing a fresh baseline, so
        # it is exempt.
        #
        # Each of the three conditions below is a way for the gate to be silently
        # disarmed rather than a way for it to pass, which is why all three fail
        # rather than being skipped:
        #
        # * No `[jit-stats]` line at all. `pyre_env` sets MAJIT_STATS=1 for every
        #   pyre run, so a run that prints none is not a bench without counters —
        #   it is a bench whose counters went missing, and reading that as
        #   "nothing to compare" skips the baseline check and the comparison
        #   together.
        # * No committed baseline. Every gate then runs against nothing and the
        #   bench prints PASS, so "never recorded" is indistinguishable from
        #   "recorded and clean" — which is how the floor came to cover 3 of 340
        #   synthetic fixtures without anyone noticing.
        # * A counter that moved, in either direction.
        # * A badness counter reading 0 because its census never ran, rather
        #   than because nothing is wrong — `JITSTATS_DENOMINATOR_FIELDS`. The
        #   other three disarms are visible as an absence (no line, no file, no
        #   move); this one is invisible, because a disarmed census and a clean
        #   one print the same digit.
        if self.args.snapshot_mode != "record":
            if jitstats is None:
                self.jitstats_absent.append(f"{backend}/{name}")
                return "fail", (
                    "no [jit-stats] line in this run's stderr — MAJIT_STATS=1 is "
                    "set for every pyre run, so every jit-stats gate below was "
                    "skipped rather than passed"
                )
            if not jitstats_path.exists():
                self.jitstats_missing.append(f"{backend}/{name}")
                return "fail", (
                    f"no committed jit-stats baseline ({jitstats_path.name}) — record it with "
                    f"`pyre/check.py --snapshot --backend {backend}`"
                )
            vacuous = _jit_stats_vacuous(stderr)
            if vacuous:
                self.jitstats_vacuous.append(f"{backend}/{name}")
                return "fail", vacuous
            regressions, improvements, banded = _jit_stats_change(
                jitstats_path.read_text(encoding="utf-8"), jitstats, jitstats_bands
            )
            if regressions or improvements:
                repeats = self._jitstats_repeats(backend, script, timeout, ungated)
                drifted = next(
                    (s for s in repeats or () if s != jitstats), None
                )
                if drifted is not None:
                    # Drift compares two samples directly and is deliberately unbanded.
                    moved = _jit_stats_change(jitstats, drifted)
                    self.jitstats_unstable.append(f"{backend}/{name}")
                    return "unstable", (
                        "jit-stats unstable — re-running the same binary moved "
                        + ", ".join(moved[0] + moved[1])
                        + ", so this run's counters are not a property of the "
                        "tree and the baseline comparison ("
                        + ", ".join(regressions + improvements)
                        + ") is not gated"
                    )
                parts = []
                if regressions:
                    parts.append("regressed: " + ", ".join(regressions))
                if improvements:
                    parts.append("improved: " + ", ".join(improvements))
                if banded:
                    parts.append("within band: " + ", ".join(banded))
                reason = (
                    "jit-stats change — " + "; ".join(parts)
                    + _jit_stats_context(jitstats)
                    + f" (re-record with `pyre/check.py --snapshot --backend {backend} "
                    f"--synthetic-pattern {Path(script).stem}`)"
                )
                if regressions:
                    self.jitstats_diffs.append(f"{backend}/{name}")
                    return "regressed", reason
                self.jitstats_improvements.append(f"{backend}/{name}")
                return "improved", reason
            if banded:
                # Assigned, not returned: every other verdict above is a
                # failure, so leaving early costs those nothing. A band means
                # the run is fine, and returning here would carry the fixture
                # past the `--snapshot-diff` and `--threshold` gates below on
                # every run of a host that sits inside its band.
                self.jitstats_banded.append(f"{backend}/{name}")
                status, reason = "banded", (
                    "jit-stats within band — " + ", ".join(banded)
                    + _jit_stats_context(jitstats)
                )

        if self.args.snapshot_mode == "record":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output, encoding="utf-8")
            time_path.write_text(f"{elapsed:.2f}", encoding="utf-8")
            if jitstats is None:
                jitstats_path.unlink(missing_ok=True)
            else:
                jitstats_path.write_text(jitstats, encoding="utf-8")

        if self.args.snapshot_mode == "diff":
            if not out_path.exists():
                self.snapshot_missing.append(f"{backend}/{name}")
            else:
                saved_out = out_path.read_text(encoding="utf-8")
                if output != saved_out:
                    self.snapshot_diffs.append(f"{backend}/{name}")
                    return "fail", "snapshot output diff"

            # The jit-stats half of this mode is gone: the gate above already
            # ran the same comparison on every run, so reaching here means it
            # found a baseline and found it equal.

        if (
            self.args.threshold is not None
            and elapsed is not None
            and elapsed != "-"
            and time_path.exists()
        ):
            saved_time_str = time_path.read_text(encoding="utf-8").strip()
            if saved_time_str and saved_time_str != "-":
                saved_time = float(saved_time_str)
                limit = saved_time * (1 + self.args.threshold / 100.0)
                if float(elapsed) > limit:
                    return "fail", f"threshold {elapsed:.2f}s > baseline {saved_time_str}s +{self.args.threshold}%"

        return status, reason

    # ── build ──

    def build_backend(self, backend):
        # `pyre-jit-trace/build.rs` compares each `build/llbc/*.ullbc` against
        # what its crate's sources hash to now, and by default reports a
        # mismatch as a `cargo::warning` — which cargo replays only when it
        # re-runs the build script, so a run whose crates were cached prints
        # nothing at all. Every number this script produces is read out of a
        # binary whose field offsets come from those artefacts, so a stale one
        # does not fail: it answers, wrongly and quietly. Four measurement runs
        # on this tree carried the mismatch and none of their logs named it.
        #
        # `PYRE_LLBC_STRICT=1` is the promotion build.rs already documents "for
        # callers that want a gate" — the same finding as `cargo::error`. The
        # cost is that a rebase which moves the LLBC crates makes the next
        # check.py stop and ask for a multi-minute re-extraction; the
        # alternative is a green run that measured the wrong bytes. Set here
        # rather than per-command so the wasm build gets it too.
        os.environ["PYRE_LLBC_STRICT"] = "1"
        cfg = CARGO_CONFIG[backend]
        if cfg.get("wasm"):
            return self.build_wasm_backend()
        print(f"Building {cfg['bin']} (release, backend={backend})...")
        cmd = [
            "cargo", "build", "--release", "-p", "pyrex",
            "--bin", cfg["bin"], *cfg["extra"],
        ]
        print("  $ " + " ".join(cmd))
        cargo_path = shutil.which("cargo") or "(not found on PATH)"
        print(f"  cargo resolved to: {cargo_path}")
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(f"ERROR: cargo build failed (exit {proc.returncode})")
            if proc.stdout:
                print("─── cargo stdout ───")
                print(proc.stdout.rstrip())
            if proc.stderr:
                print("─── cargo stderr ───")
                print(proc.stderr.rstrip())
            print("────────────────────")
            cargo_output = (proc.stderr or "") + (proc.stdout or "")
            if "LLBC STALE" in cargo_output:
                # `PYRE_LLBC_STRICT=1` above turned build.rs's staleness
                # warning into the build failure that got us here. It already
                # printed the exact `extract-llbc.py` line naming the crates
                # that moved, so repeat the reason rather than the command.
                print(red("LLBC artefacts under build/llbc/ are STALE."))
                print("Field offsets come from them, so a run on this tree "
                      "would measure the wrong bytes.")
                print("Re-extract with the command build.rs printed above, "
                      "then re-run this script.")
                sys.exit(1)
            llbc_missing = (
                # translator runtime panic (majit-translate/src/lib.rs)
                "no LLBC source resolved" in cargo_output
                # pyre-jit-trace/build.rs compile-time preflight
                or "Extract the LLBC" in cargo_output
            )
            if llbc_missing:
                # The JIT front-end has no MIR to lower because build/llbc/
                # is empty. This is a setup step, not a toolchain fault — the
                # rustup diagnostics below would be noise, so point at the
                # producer instead.
                print(red("LLBC artefacts are missing under build/llbc/."))
                print("Run the extractor first, then re-run this script:")
                # No crate arguments: extract the full DEFAULT_CRATES set
                # (pyre-object, pyre-interpreter, pyre-jit). The exact
                # eval::eval_loop_jit portal lives in pyre-jit.ullbc, so a
                # production JIT build requires all three artifacts.
                print("    scripts/extract-llbc.py")
            else:
                self._print_cargo_diagnostics(cargo_path)
            sys.exit(1)
        lines = (proc.stdout or "").strip().splitlines() + (proc.stderr or "").strip().splitlines()
        if lines:
            print(lines[-1])

    def build_wasm_backend(self):
        """Build the wasm32 `pyre-wasm` module and the native `pyre-wasm-runner`.

        The runner loads the module (via `$PYRE_WASM_MODULE`, defaulting to
        `WASM_MODULE_PATH`) and executes it under wasmtime, so two artefacts are
        produced: the wasm module (needs the export-table / custom-getrandom
        flags) and the host runner binary.
        """
        steps = [
            (
                "pyre-wasm (wasm32, --features wasm-host)",
                [
                    "cargo", *WASM_CARGO_TOOLCHAIN, "build", "--release",
                    "-p", "pyre-wasm",
                    "--target", "wasm32-unknown-unknown",
                    "--no-default-features", "--features", "wasm-host",
                    *WASM_BUILD_STD_FLAGS,
                ],
                {
                    **os.environ,
                    "RUSTFLAGS": WASM_RUSTFLAGS,
                },
            ),
            (
                "pyre-wasm-runner (native wasmtime host)",
                ["cargo", "build", "--release", "-p", "pyre-wasm-runner"],
                None,
            ),
        ]
        for label, cmd, env in steps:
            print(f"Building {label}...")
            print("  $ " + " ".join(cmd))
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", env=env,
            )
            if proc.returncode != 0:
                print(f"ERROR: cargo build failed (exit {proc.returncode})")
                if proc.stdout:
                    print("─── cargo stdout ───")
                    print(proc.stdout.rstrip())
                if proc.stderr:
                    print("─── cargo stderr ───")
                    print(proc.stderr.rstrip())
                print("────────────────────")
                sys.exit(1)
            lines = (proc.stderr or "").strip().splitlines()
            if lines:
                print(lines[-1])
        if not Path(WASM_BUILD_OUTPUT).exists():
            print(f"ERROR: wasm module not produced at {WASM_BUILD_OUTPUT}")
            sys.exit(1)
        # Snapshot the wasm-host build to a stable path so a later `web` build of
        # the same crate cannot overwrite the module the runner loads. Copy when
        # the bytes actually changed: rewriting an identical file would bump its
        # mtime, and the runner's `<module>.cwasm` compiled cache is keyed by
        # the module's content hash, so an identical rewrite buys nothing.
        src_bytes = Path(WASM_BUILD_OUTPUT).read_bytes()
        dst = Path(WASM_MODULE_PATH)
        if not dst.exists() or dst.read_bytes() != src_bytes:
            dst.write_bytes(src_bytes)

        if WASM_ENGINE == "wasmtime":
            self._warm_wasm_cache()

    def _warm_wasm_cache(self):
        """Compile the wasmtime `.cwasm` cache once here, untimed.

        wasmtime recompiles the whole ~14MB module (~5s) on a cold start. The
        runner caches that compilation in `<module>.cwasm`, but the wasm build
        is non-deterministic, so each rebuild yields a fresh module that
        invalidates the cache. Warming it in the build phase moves that fixed
        cost out of every measured benchmark (including the first), so the
        reported times reflect Python execution, not module compilation.

        Fuel metering emits a different module, cached separately as
        `<module>.fuel.cwasm`, so warm that one too: the wasm runtime codegen
        regression tests all run under PYRE_WASM_JIT_STATS, and they start six
        runners at once — without this every one of them recompiles.
        """
        runner = default_binary("wasm")
        if not Path(runner).exists():
            return
        env = dict(os.environ)
        env["PYRE_WASM_MODULE"] = str(Path(WASM_MODULE_PATH).resolve())
        env["PYRE_WASM_ENGINE"] = "wasmtime"
        print("Warming wasmtime module cache (.cwasm)...")
        for extra in ({}, {"PYRE_WASM_JIT_STATS": "1"}):
            try:
                subprocess.run(
                    [runner, "--engine", "wasmtime", os.devnull],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env={**env, **extra}, timeout=120,
                )
            except (OSError, subprocess.SubprocessError):
                pass  # best-effort; a cold first bench just pays the compile once

    def _print_cargo_diagnostics(self, cargo_path):
        """Dump the toolchain state when cargo refuses to run.

        Targets the macOS CI failure mode where Swatinem/rust-cache restores
        a stale `~/.cargo/bin/cargo` rustup proxy after a runner-image bump.
        The new rustup's state under `~/.rustup/` then mismatches the old
        proxy, which falls back to rustup-init mode and clap-rejects `build`.
        """
        print("─── cargo diagnostics ───")
        print(f"PATH = {os.environ.get('PATH', '')}")
        print(f"CARGO_HOME = {os.environ.get('CARGO_HOME', '(unset)')}")
        print(f"RUSTUP_HOME = {os.environ.get('RUSTUP_HOME', '(unset)')}")
        if sys.platform != "win32":
            for which_cmd in (["which", "-a", "cargo"], ["which", "-a", "rustup"]):
                self._run_diag(which_cmd)
            cargo_dir = os.path.dirname(cargo_path) if os.path.sep in cargo_path else ""
            if cargo_dir:
                self._run_diag(["ls", "-laHi", cargo_dir])
            if cargo_dir and os.path.isfile(cargo_path):
                self._run_diag(["file", cargo_path])
        self._run_diag(["cargo", "--version"])
        self._run_diag(["rustup", "show"])
        print("─────────────────────────")

    @staticmethod
    def _run_diag(cmd):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            print(f"  $ {' '.join(cmd)}\n    [error: {e}]")
            return
        out = (proc.stdout or "") + (proc.stderr or "")
        print(f"  $ {' '.join(cmd)} (exit {proc.returncode})")
        for line in out.rstrip().splitlines():
            print(f"    {line}")

    # ── warmup ──

    def warmup(self, script):
        sys.stdout.write(f"  {'warmup':<10s}")
        sys.stdout.flush()
        for runner in [PYTHON3, PYPY3]:
            try:
                subprocess.run(
                    [runner, script],
                    stdout=subprocess.DEVNULL,
                    timeout=30,
                )
            except Exception:
                pass
        for backend in ALL_BACKENDS:
            if self.enabled(backend):
                try:
                    subprocess.run(
                        [self._pyre(backend), script],
                        stdout=subprocess.DEVNULL,
                        timeout=30,
                        env=pyre_env(),
                    )
                except Exception:
                    pass
        print(dim("done"))

    # ── single-backend bench run ──

    def _retry_performance_gate(
        self, backend, script, timeout, baseline_cmd, expected_output,
        baseline_key,
    ):
        """Return median (pyre, baseline) timings after a slow first sample.

        A retry is deliberately limited to the failed gate: correctness has
        already passed, and a nonzero exit, changed output, or JIT panic in any
        retry remains a failure rather than being hidden by a median.
        """
        attempts = PERF_RETRY_RUNS
        effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))
        # The wasm gate's baseline is a pyre binary, not an oracle, so it needs
        # the pyre environment the numerator already runs under — otherwise the
        # ratio compares a pinned run against an unpinned one. It was also the
        # one pyre spawn left outside `pyre_env()`, which is how a bytecode
        # cache still appeared under a run that pins PYTHONDONTWRITEBYTECODE.
        baseline_is_pyre = baseline_key in ALL_BACKENDS
        baseline_env = pyre_env() if baseline_is_pyre else None
        pyre_times = []
        baseline_times = []
        for _ in range(attempts):
            baseline_output, baseline_time, baseline_code, baseline_stderr = run_timed(
                baseline_cmd, timeout_s=effective_timeout, env=baseline_env,
            )
            if baseline_code != 0:
                return None
            # A pyre baseline is under test itself, so it answers to the same
            # bar as the numerator: a median built on a denominator that
            # printed the wrong answer or panicked out of the JIT would let the
            # failing gate pass on a time no correct run produced.
            if baseline_is_pyre and (
                baseline_output != expected_output
                or _jit_panic_reason(baseline_stderr)
            ):
                return None
            output, elapsed, code, stderr = run_timed(
                [self._pyre(backend), script], timeout_s=effective_timeout,
                env=pyre_env(),
            )
            if (
                code != 0
                or output != expected_output
                or _jit_panic_reason(stderr)
            ):
                return None
            baseline_times.append(baseline_time)
            pyre_times.append(elapsed)
        return statistics.median(pyre_times), statistics.median(baseline_times)

    def _baseline_exec_time_clamped(self, baseline, baseline_time):
        """Whether startup subtraction pinned a baseline to its floor."""
        exec_b = self._exec_time(baseline, baseline_time)
        return (
            not self.args.no_startup_subtract
            and exec_b not in (None, "-")
            and float(exec_b) <= EXEC_TIME_FLOOR_S
        )

    def _baseline_exec_time_thin(self, baseline, baseline_time):
        """Whether a baseline clears the floor but not the floor gate's bar.

        Reporting only: the ceiling accepts any value above
        `EXEC_TIME_FLOOR_S`, while the floor gate requires
        `FLOOR_GATE_MIN_BASELINE_S`. Mark ratios in the intervening band so
        they are not mistaken for values accepted by both directions.
        """
        exec_b = self._exec_time(baseline, baseline_time)
        return (
            not self.args.no_startup_subtract
            and exec_b not in (None, "-")
            and EXEC_TIME_FLOOR_S < float(exec_b) < FLOOR_GATE_MIN_BASELINE_S
        )

    def _performance_gate_passed(
        self, backend, script, timeout, elapsed, limit, baseline_time,
        baseline_cmd, expected_output, baseline_key, minimum=None,
    ):
        """Check one performance ratio, retrying a failure by median.

        The pass/fail decision compares execution-only (startup-subtracted)
        times; the returned elapsed/baseline stay raw for reporting. The
        returned bound is ``ceiling`` or ``floor`` when the gate fails.
        """
        # Each execution-only value is a difference between two independently
        # measured quantities (bench - empty-program startup).  On Windows each
        # quantity may be off by one scheduler tick, so the comparison
        # ``pyre <= baseline * limit`` needs 2q on the left and 2q*limit on the
        # right.  Adding one tick to every raw sample cannot model this: those
        # additions cancel during startup subtraction.
        #
        # The right-hand ``baseline * limit`` amplifies the baseline's own
        # measurement error by `limit`, and off Windows that term was missing:
        # the buffer carried only the left-hand `BENCH_COMPARE_BUFFER_S` and no
        # allowance for a noisy denominator.  A baseline near EXEC_TIME_FLOOR_S
        # is a difference of two medians each granular to about one floor, so its
        # residual run-to-run error is ~EXEC_TIME_FLOOR_S, and the budget it sets
        # swings by `limit * EXEC_TIME_FLOOR_S`.  This makes the effective
        # ceiling `limit * (1 + EXEC_TIME_FLOOR_S / baseline)`: the grace is
        # `floor / baseline`, real only when the baseline is close to the floor
        # -- exactly where the denominator cannot be trusted -- and vanishing
        # for a baseline that is real work.  `exception_traceback_loop_forms`
        # read 35.6x on one runner and 37.1x on another against an unchanged
        # ~2.3s exec, a 0.06s baseline wobbling +-0.004s across the 36x line.
        compare_buffer = BENCH_COMPARE_BUFFER_S
        if sys.platform == "win32":
            compare_buffer += 2 * WIN_TIMER_QUANTUM_S * (1 + limit)
        else:
            compare_buffer += limit * EXEC_TIME_FLOOR_S

        def failed_bound(measured, baseline_value):
            exec_measured = self._exec_time(backend, measured)
            exec_baseline = self._exec_time(baseline_key, baseline_value)
            # A baseline pinned to EXEC_TIME_FLOOR_S by startup subtraction is
            # not a measurement of the baseline, it is the floor constant. The
            # ratio built on it is `exec_measured / EXEC_TIME_FLOOR_S`, so the
            # recorded ceiling it is compared against is an absolute wall-clock
            # budget of `limit * EXEC_TIME_FLOOR_S` seconds -- and that budget
            # was fitted on whichever host last wrote the header. Applying it
            # elsewhere compares two hosts' wall clocks with no baseline
            # standing between them, which is what the printed line already
            # says ("ratio not a measurement") while failing the run on it.
            # Measured: `class_reassign_hot` read 27.0x and 27.3x on one host
            # and 49.2x on a CI runner against the same code, because only the
            # numerator moves.
            #
            # The earlier reading kept the ceiling armed here because
            # measured/floor is a LOWER bound on the true ratio, so a failure
            # does prove the fixture is at least that many times slower than
            # pypy. It is still not a bound this ceiling can judge: the
            # recorded number carries the same clamp, so no pypy measurement
            # enters the comparison on either side. Three consecutive `main`
            # runs failed exactly this way on three different fixtures and two
            # runners -- global_cell_shortpreamble_hot 24.1x > 19x,
            # class_reassign_hot 49.2x > 47x, reentrant_key_eq_mutation 10.3x >
            # 5x -- which is a population, not three regressions.
            #
            # Only the ceiling relaxes: the floor declines to arm below
            # FLOOR_GATE_MIN_BASELINE_S for the neighbouring reason, which a
            # clamped baseline is always under. The number stays visible either
            # way -- the comparison table prints it with a `~`. A fixture that
            # wants a ratio gate has to give pypy enough work to measure.
            if self._baseline_exec_time_clamped(baseline_key, baseline_value):
                return None
            if exec_measured > exec_baseline * limit + compare_buffer:
                return "ceiling"
            if (
                minimum is not None
                and exec_baseline >= FLOOR_GATE_MIN_BASELINE_S
                and exec_measured * (limit / minimum) + compare_buffer
                < exec_baseline * limit
            ):
                return "floor"
            return None

        bound = failed_bound(elapsed, baseline_time)
        if bound is None:
            return True, None, elapsed, baseline_time, ""

        retry = self._retry_performance_gate(
            backend, script, timeout, baseline_cmd, expected_output,
            baseline_key,
        )
        if retry is not None:
            median_elapsed, median_baseline = retry
            bound = failed_bound(median_elapsed, median_baseline)
            return (
                bound is None, bound, median_elapsed, median_baseline,
                f"median {PERF_RETRY_RUNS}",
            )

        return False, bound, elapsed, baseline_time, ""

    def _gate_fail_detail(
        self, backend, baseline, measured, baseline_time, limit, bound,
        minimum=None,
    ):
        """One-line FAIL detail using the exact numbers the gate compared.

        The gate decides on startup-subtracted exec times
        rather than raw run times, so those exec times are printed alongside
        their true measured ratio and the failed gate threshold. Every number
        on the line is arithmetically self-consistent: exec_measured /
        exec_baseline equals the shown ratio, which is outside the shown gate.
        """
        exec_m = self._exec_time(backend, measured)
        exec_b = self._exec_time(baseline, baseline_time)
        if exec_b in (None, "-") or float(exec_b) <= 0:
            ratio = "-"
        else:
            ratio = f"{float(exec_m) / float(exec_b):.1f}x"
        if bound == "floor":
            detail = (
                f"exec {exec_m:.2f}s vs {baseline} {exec_b:.2f}s  "
                f"ratio {ratio} < gate {float(minimum):g}x; "
                f"max-pypy-ratio ceiling {float(limit):g}x needs re-recording"
            )
        else:
            detail = (
                f"exec {exec_m:.2f}s > {baseline} {exec_b:.2f}s  "
                f"ratio {ratio} > gate {float(limit):g}x"
            )
        return detail

    def _run_backend_bench(
        self, backend, name, script, timeout,
        vs_cpython, vs_pypy, t_cpython, t_pypy, pypy_output,
        wasm_float_tol=False, max_rss_mb=None,
    ):
        pyre_bin = self._pyre(backend)
        effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))

        sys.stdout.write(f"    {backend:<10s}")
        sys.stdout.flush()

        output, elapsed, code, stderr = run_timed(
            [pyre_bin, script], timeout_s=effective_timeout, env=pyre_env(),
        )
        # Read before any gate re-runs the fixture and overwrites the slot.
        peak_rss_mb = last_run_peak_rss_mb()

        panic_reason = _jit_panic_reason(stderr)
        if panic_reason:
            self._record(backend, False, name, panic_reason)
            print(f"{red('JIT-PANIC')}  {panic_reason}")
            self._append_comparison(backend, name, t_cpython, t_pypy, "FAIL")
            return

        if code != 0:
            if code == 124:
                self._record(backend, False, name, f"timeout (>{effective_timeout}s)")
                print(f"{red('TIMEOUT')}  >{effective_timeout}s")
            else:
                self._record(backend, False, name, f"crash (exit {code})")
                print(f"{red('CRASH')} (exit {code})")
            self._append_comparison(backend, name, t_cpython, t_pypy, "FAIL")
            return

        # Every backend requires byte-identical output. The sole exception is a
        # bench explicitly marked wasm_float_tol=True AND run on the wasm
        # backend: wasm32's scalar ISA cannot reproduce the platform libm's
        # arch-tuned pow, so its transcendental-heavy output drifts by a bounded
        # amount (see `wasm_outputs_match`). This is opt-in per bench, never a
        # blanket wasm allowance — every other wasm bench stays byte-exact.
        if backend == "wasm" and wasm_float_tol:
            matched = wasm_outputs_match(output, pypy_output)
        else:
            matched = output == pypy_output
        if not matched:
            self._record(backend, False, name, "wrong output")
            print(f"{red('WRONG')}")
            _dump_output_mismatch(output, pypy_output)
            self._append_comparison(backend, name, t_cpython, t_pypy, "WRONG")
            return

        # Recorded before the gates so a later median retry cannot make one
        # backend's stored time depend on which gates its own run happened to
        # trip. dynasm runs before wasm in ALL_BACKENDS, so the wasm gate below
        # finds this fixture's dynasm time already in place.
        self.bench_elapsed[(backend, name)] = elapsed

        def _ratio(elapsed_val, pypy_val):
            num = self._exec_time(backend, elapsed_val)
            den = self._exec_time("pypy", pypy_val)
            if den in (None, "-") or float(den) <= 0 or num in (None, "-"):
                return "-"
            # Display only: neither marker reaches a verdict. `~` says no gate
            # was applied; `?` says one was, against a baseline under the bar
            # the floor gate uses. An unmarked ratio now means both directions
            # of the gate accepted the denominator, which is what a reader has
            # been entitled to assume all along.
            if self._baseline_exec_time_clamped("pypy", pypy_val):
                marker = "~"
            elif self._baseline_exec_time_thin("pypy", pypy_val):
                marker = "?"
            else:
                marker = ""
            return f"{marker}{float(num) / float(den):.1f}x"

        ratio = _ratio(elapsed, t_pypy) if elapsed > 0 else "-"

        # Ratio and memory failures do not invalidate the run's counters, so
        # keep comparing or recording the jit-stats baseline after they fail.
        failures = []
        retry_note = ""
        if vs_cpython and t_cpython not in (None, "-"):
            passed, bound, checked_elapsed, checked_baseline, retry_note = self._performance_gate_passed(
                backend, script, timeout, elapsed, vs_cpython, float(t_cpython),
                [PYTHON3, script], pypy_output, "cpython",
            )
            if not passed:
                detail = self._gate_fail_detail(
                    backend, "cpython", checked_elapsed, checked_baseline,
                    vs_cpython, bound,
                )
                suffix = f" ({retry_note})" if retry_note else ""
                failures.append(
                    (
                        f"{red('SLOWER')}  pyre {detail}{suffix}",
                        detail,
                        fmt_time(f"{elapsed:.2f}"),
                        f"({ratio} vs pypy)",
                    )
                )
            if passed and retry_note:
                elapsed = checked_elapsed
                ratio = _ratio(elapsed, t_pypy)

        if vs_pypy and t_pypy not in (None, "-"):
            minimum = perf_gate_floor(vs_pypy)
            passed, bound, checked_elapsed, checked_baseline, retry_note = self._performance_gate_passed(
                backend, script, timeout, elapsed, vs_pypy, float(t_pypy),
                [PYPY3, script], pypy_output, "pypy", minimum,
            )
            if not passed:
                detail = self._gate_fail_detail(
                    backend, "pypy", checked_elapsed, checked_baseline,
                    vs_pypy, bound, minimum,
                )
                suffix = f" ({retry_note})" if retry_note else ""
                label = "FASTER" if bound == "floor" else "SLOWER"
                failures.append(
                    (
                        f"{red(label)}  pyre {detail}{suffix}",
                        detail,
                        fmt_time(f"{elapsed:.2f}"),
                        f"({ratio} vs pypy)",
                    )
                )
            if passed and retry_note:
                elapsed = checked_elapsed
                t_pypy = checked_baseline
                ratio = _ratio(elapsed, t_pypy)

        # `# pyre-check: max-rss-mb=` — the axis the ratio gates cannot see.
        # Checked after them so a fixture that is both slow and fat reports the
        # slowness first, while still reporting both failures.
        if max_rss_mb is not None and peak_rss_mb is not None and peak_rss_mb > max_rss_mb:
            detail = f"peak RSS {peak_rss_mb:.0f}MB > {max_rss_mb:.0f}MB"
            failures.append(
                (
                    f"{red('FATTER')}  pyre {detail}",
                    detail,
                    fmt_time(f"{elapsed:.2f}"),
                    f"({ratio} vs pypy)",
                )
            )

        # wasm carries no tuned per-fixture ratio, so it gates against dynasm's
        # execution-only time from this same invocation instead. Placed after
        # the other ratio gates so a fixture that is slow against several
        # baselines still reports each one.
        if backend == "wasm":
            dynasm_elapsed = self.bench_elapsed.get(("dynasm", name))
            dynasm_exec = (
                self._exec_time("dynasm", dynasm_elapsed)
                if dynasm_elapsed is not None
                else None
            )
            # Both sides of this ratio are measured in this invocation, so the
            # denominator carries the full startup-subtraction error rather
            # than a recorded constant's stability. `_baseline_exec_time_clamped`
            # would only reject one already pinned to EXEC_TIME_FLOOR_S, which
            # leaves the band up to FLOOR_GATE_MIN_BASELINE_S dividing by
            # something the same size as its own error.
            ceiling = wasm_ratio_gate(script) or WASM_MAX_DYNASM_RATIO
            if ceiling > WASM_MAX_DYNASM_RATIO:
                self.wasm_ratio_allowed.append((name, ceiling))
            if dynasm_exec in (None, "-") or (
                float(dynasm_exec) < FLOOR_GATE_MIN_BASELINE_S
            ):
                self.wasm_ratio_ungated.append(name)
            else:
                passed, bound, checked_elapsed, checked_baseline, retry_note = (
                    self._performance_gate_passed(
                        backend, script, timeout, elapsed,
                        ceiling, dynasm_elapsed,
                        [self._pyre("dynasm"), script], pypy_output, "dynasm",
                    )
                )
                if not passed:
                    detail = self._gate_fail_detail(
                        backend, "dynasm", checked_elapsed, checked_baseline,
                        ceiling, bound,
                    )
                    suffix = f" ({retry_note})" if retry_note else ""
                    failures.append(
                        (
                            f"{red('SLOWER')}  pyre {detail}{suffix}",
                            detail,
                            fmt_time(f"{elapsed:.2f}"),
                            f"({ratio} vs pypy)",
                        )
                    )

        snap_status, snap_reason = self._apply_snapshot_gate(
            backend, name, script, output, stderr, elapsed, timeout,
        )
        if snap_status != "ok":
            if snap_status in ("unstable", "banded"):
                # Warned, not failed: unstable counters cannot be gated, while
                # banded counters stayed inside their declared allowance.
                note_label = snap_status.upper()
                note_line = f"{yellow(note_label)}  {snap_reason}"
            else:
                # `improved` is still a failure; the label only tells the
                # reader whether to investigate or just re-record.
                label = "IMPROVED" if snap_status == "improved" else "SNAPDIFF"
                paint = yellow if snap_status == "improved" else red
                failures.append(
                    (f"{paint(label)}  {snap_reason}", snap_reason, label, None)
                )
                note_line = None
        else:
            note_line = None

        if failures:
            self._record(
                backend, False, name,
                "; ".join(detail for _, detail, _, _ in failures),
            )
            for index, (line, _, _, _) in enumerate(failures):
                print(f"{' ' * 14 if index else ''}{line}")
            if note_line is not None:
                print(f"{' ' * 14}{note_line}")
            _, _, comparison_cell, comparison_note = failures[0]
            self._append_comparison(
                backend, name, t_cpython, t_pypy,
                comparison_cell, comparison_note,
            )
            return

        if note_line is not None:
            self._record(backend, True, name, f"{elapsed:.2f}s")
            print(note_line)
            self._append_comparison(backend, name, t_cpython, t_pypy, note_label)
            return

        self._record(backend, True, name, f"{elapsed:.2f}s")
        suffix = f" ({retry_note})" if retry_note else ""
        print(f"{green('PASS')}  {elapsed:.2f}s{suffix}")
        self._append_comparison(
            backend, name, t_cpython, t_pypy,
            fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
        )

    # ── top-level bench entry ──

    def run_bench(
        self, name, script, timeout,
        dynasm_vs_cpython=None, dynasm_vs_pypy=None,
        cranelift_vs_cpython=None, cranelift_vs_pypy=None,
        skip_backends=(), *, wasm_float_tol=False,
    ):
        """Run one benchmark on each enabled backend."""
        need_cpython = False
        if (
            self.enabled("dynasm")
            and "dynasm" not in skip_backends
            and dynasm_vs_cpython
        ):
            need_cpython = True
        if (
            self.enabled("cranelift")
            and "cranelift" not in skip_backends
            and cranelift_vs_cpython
        ):
            need_cpython = True
        if self.args.full:
            need_cpython = True

        print(f"  {name}")

        t_cpython = "-"
        cpython_code = 0
        if need_cpython:
            sys.stdout.write(f"    {'cpython':<10s}")
            sys.stdout.flush()
            cpython_output, t_cpu, cpython_code, _ = run_timed([PYTHON3, script])
            t_cpython = t_cpu
            if cpython_code != 0:
                print(f"{red('CRASH')} (exit {cpython_code})")
            else:
                print(f"{dim('done')}  {t_cpython:.2f}s")

        sys.stdout.write(f"    {'pypy':<10s}")
        sys.stdout.flush()
        pypy_output, pypy_cpu, pypy_code, _ = run_timed([PYPY3, script])
        t_pypy = pypy_cpu if pypy_code == 0 else "-"
        if pypy_code != 0:
            print(f"{red('CRASH')} (exit {pypy_code})")
            for backend in ALL_BACKENDS:
                if self.enabled(backend):
                    self._record(backend, False, name, "pypy crash")
                    self._append_comparison(backend, name, t_cpython, "-", "FAIL")
            return
        print(f"{dim('done')}  {t_pypy:.2f}s")

        for backend, vs_cpython, vs_pypy in [
            ("dynasm", dynasm_vs_cpython, dynasm_vs_pypy),
            ("cranelift", cranelift_vs_cpython, cranelift_vs_pypy),
            ("wasm", None, None),
        ]:
            if not self.enabled(backend):
                continue
            if backend in skip_backends:
                sys.stdout.write(f"    {backend:<10s}")
                print(dim("skip"))
                self._append_comparison(
                    backend, name, t_cpython, t_pypy, "skip",
                )
                continue
            if vs_cpython and cpython_code != 0:
                sys.stdout.write(f"    {backend:<10s}")
                print(f"{red('FAIL')}  missing cpython baseline")
                self._record(backend, False, name, "cpython crash")
                self._append_comparison(backend, name, "-", t_pypy, "FAIL")
                continue
            self._run_backend_bench(
                backend, name, script, timeout,
                vs_cpython, vs_pypy, t_cpython, t_pypy, pypy_output,
                wasm_float_tol=wasm_float_tol,
            )

    # ── self-checking regression guard ──

    def run_selfcheck(self, name, script, timeout, expect="PASS", skip_backends=()):
        """Run a self-checking regression script on each enabled backend.

        The script asserts its own invariant (exit 0 AND prints *expect*);
        check.py only honors the exit code and the required marker. Used for
        guards whose signal is an asserted invariant, not byte-identical
        output, so they cannot go through `run_bench` or synthetic parity.

        *skip_backends* names backends the guard does not apply to (e.g. a
        `time`-module timing guard cannot run on the wasm guest, which has no
        `time` module).
        """
        print(f"  {name}")
        for backend in ALL_BACKENDS:
            if not self.enabled(backend):
                continue
            if backend in skip_backends:
                sys.stdout.write(f"    {backend:<10s}")
                print(dim("skip"))
                self._append_comparison(backend, name, "-", "-", "skip")
                continue
            effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))
            sys.stdout.write(f"    {backend:<10s}")
            sys.stdout.flush()
            output, elapsed, code, stderr = run_timed(
                [self._pyre(backend), script],
                timeout_s=effective_timeout,
                env=pyre_env(),
            )
            panic_reason = _jit_panic_reason(stderr)
            if panic_reason:
                self._record(backend, False, name, panic_reason)
                print(f"{red('JIT-PANIC')}  {panic_reason}")
                self._append_comparison(backend, name, "-", "-", "FAIL")
                continue
            if code != 0:
                detail = (
                    f"timeout (>{effective_timeout}s)" if code == 124
                    else f"exit {code}"
                )
                self._record(backend, False, name, detail)
                print(f"{red('FAIL')}  {detail}")
                _dump_failed_run(output, stderr)
                self._append_comparison(backend, name, "-", "-", "FAIL")
                continue
            if expect not in output:
                self._record(backend, False, name, f"missing '{expect}'")
                print(f"{red('FAIL')}  missing '{expect}'")
                _dump_failed_run(output, stderr)
                self._append_comparison(backend, name, "-", "-", "FAIL")
                continue
            self._record(backend, True, name, "")
            print(f"{green('PASS')}  {elapsed:.2f}s")
            self._append_comparison(backend, name, "-", "-", f"{elapsed:.2f}s")

    # ── vendored CPython suite ──

    def run_cpython_suite(self):
        """Gate the vendored CPython test suite against its recorded baseline.

        Every other stage in this file runs code we wrote, so it can only
        catch a defect in a shape somebody already thought of.  This one runs
        CPython's own tests, and it is the stage that reports a JIT miscompile
        nobody wrote a fixture for: `test.test_datetime` failed on a
        specialised-pair subscript fold while the whole synthetic corpus and
        every parity fixture stayed green.

        The baseline records one verdict per module per backend and follows
        the host: `run.py` reads a `baseline.<platform>-<machine>.json`
        overlay before the shared file, so a verdict a host disagrees with
        (dynasm's codegen is arch-specific) is recorded there rather than
        making the stage unusable off one machine.

        Off by default and reached only through `--cpython-suite`: the suite
        costs more wall time than every other stage here put together, and the
        `cpython-tests` CI job already runs it on its own schedule.  Pass the
        flag locally when a change could move a verdict the synthetic corpus
        does not cover.
        """
        name = "cpython-suite"
        backend = "dynasm"
        print(f"  {name}")
        if not self.enabled(backend):
            sys.stdout.write(f"    {backend:<10s}")
            print(dim("skip (backend not enabled)"))
            self._append_comparison(backend, name, "-", "-", "skip")
            return
        sys.stdout.write(f"    {backend:<10s}")
        sys.stdout.flush()
        output, elapsed, code, stderr = run_timed(
            [
                PYTHON3, str(Path(__file__).parent / "cpython_tests" / "run.py"),
                "--binary", str(self._pyre(backend)),
                "--baseline", str(CPYTHON_SUITE_BASELINE),
                "--jobs", str(max(1, (os.cpu_count() or 4) - 1)),
                "--timeout", str(CPYTHON_SUITE_MODULE_TIMEOUT_S),
            ],
            timeout_s=CPYTHON_SUITE_TIMEOUT_S,
            env=pyre_env(),
        )
        # `N to run,` rather than the runner's own "no regressions": a
        # selection that came out empty prints every counter as zero and
        # still exits 0, which reads as a pass and tests nothing.
        #
        # The default runner gates only modules whose recorded baseline passes.
        # Report skipped and non-gated counts so this subset is not mistaken
        # for the full suite.
        selected = re.search(
            r"^(\d+) to run, (\d+) skipped(?:, (\d+) not gated \(non-PASS\))?,",
            output,
            re.M,
        )
        if code == 124:
            detail = f"timeout (>{CPYTHON_SUITE_TIMEOUT_S}s)"
        elif selected is None:
            detail = "runner printed no selection line"
        elif selected.group(1) == "0":
            detail = "selected 0 modules"
        elif code != 0:
            regressions = re.findall(r"^  - (\S+): (.*)$", output, re.M)
            detail = "; ".join(f"{m} {d}" for m, d in regressions) or f"exit {code}"
        else:
            detail = ""
        if detail:
            self._record(backend, False, name, detail)
            print(f"{red('FAIL')}  {detail}")
            _dump_failed_run(output, stderr)
            self._append_comparison(backend, name, "-", "-", "FAIL")
            return
        self._record(backend, True, name, "")
        ran, not_run, degated = (
            selected.group(1), selected.group(2), selected.group(3) or "0",
        )
        print(
            f"{green('PASS')}  {ran} gated, {degated} not gated, "
            f"{not_run} skipped, {elapsed:.0f}s"
        )
        self._append_comparison(backend, name, "-", "-", f"{elapsed:.0f}s")

    # ── synthetic parity suite ──

    def run_synthetic_bench(self, path, timeout):
        name = f"synth/{Path(path).stem}"
        effective_timeout = scaled_timeout(timeout, self.args.timeout_scale)
        try:
            max_pypy_ratio = synth_perf_gate(path)
            max_rss_mb = synth_rss_gate(path)
            skip_backends = synth_skip_backends(path)
            skip_cpython = synth_skip_cpython(path)
            no_cpython = synth_no_cpython(path)
        except ValueError as e:
            print(f"{red('ERROR')}: {e}")
            sys.exit(1)

        print(f"  {name}")

        sys.stdout.write(f"    {'cpython':<10s}")
        sys.stdout.flush()
        if skip_cpython or no_cpython:
            # `# pyre-check: skip-cpython` — the fixture is sized past what
            # cpython can usefully run, so running it would spend the whole
            # reference timeout to arrive at the same drop.
            # `# pyre-check: no-cpython` — the fixture's subject is a place
            # pyre follows pypy and the reference does something else, so the
            # reference is not an oracle here. Same shape as the timeout arm
            # below: `cpython_output = None` leaves pypy as the only baseline
            # the backends are compared against.
            cpython_output = None
            # Spelled the same way as the timeout skip below, so the comparison
            # table's cpython column reads the same for every kind of skip.
            t_cpython = "skip (pypy oracle)" if no_cpython else "skip (opt)"
            print(dim(t_cpython))
            self.cpython_declared_skips.append(name)
        else:
            cpython_output, cpython_time, cpython_code, _ = run_timed(
                [PYTHON3, path], timeout_s=SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S,
            )
            if cpython_code == 124:
                # Not a crash: the fixture is sized for the JITs and cpython is
                # only the reference. Drop it and let pypy be the baseline the
                # backends are compared against, rather than spend the fixture's
                # whole timeout on an interpreter nothing is gated on.
                cpython_output = None
                t_cpython = f"skip (>{SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S:g}s)"
                print(dim(t_cpython))
                self.cpython_reference_skips.append(name)
            elif cpython_code != 0:
                print(f"{red('CRASH')} (exit {cpython_code})")
                for backend in ALL_BACKENDS:
                    if self.enabled(backend):
                        self._record(backend, False, name, "cpython crash")
                        self._append_comparison(backend, name, "-", "-", "FAIL")
                return
            else:
                t_cpython = cpython_time
                print(f"{dim('done')}  {t_cpython:.2f}s")

        sys.stdout.write(f"    {'pypy':<10s}")
        sys.stdout.flush()
        pypy_output, pypy_time, pypy_code, _ = run_timed(
            [PYPY3, path], timeout_s=effective_timeout,
        )
        if pypy_code != 0:
            print(f"{red('CRASH')} (exit {pypy_code})")
            for backend in ALL_BACKENDS:
                if self.enabled(backend):
                    self._record(backend, False, name, "pypy crash")
                    self._append_comparison(backend, name, t_cpython, "-", "FAIL")
            return
        t_pypy = pypy_time
        print(f"{dim('done')}  {t_pypy:.2f}s")

        if cpython_output is not None and cpython_output != pypy_output:
            print(f"    {'baseline':<10s}{red('WRONG')}  cpython output differs from pypy")
            for backend in ALL_BACKENDS:
                if self.enabled(backend):
                    self._record(backend, False, name, "cpython/pypy output mismatch")
                    self._append_comparison(
                        backend, name, t_cpython, t_pypy, "BASEFAIL",
                    )
            return

        for backend in ALL_BACKENDS:
            if not self.enabled(backend):
                continue
            # `# pyre-check: skip-backends=` — the fixture guards a mechanism
            # this backend does not implement, so running it here would only
            # restate that gap.
            if backend in skip_backends:
                sys.stdout.write(f"    {backend:<10s}")
                print(dim("skip"))
                self._append_comparison(backend, name, t_cpython, t_pypy, "skip")
                continue
            # wasm carries no perf-ratio gate, matching `run_bench` (which has no
            # `wasm_vs_*` parameter at all): it legitimately runs a few× slower
            # than the native backends a fixture's ratio is tuned for, so one
            # ratio cannot gate both — see WASM_TIMEOUT_SCALE. Its per-bench
            # timeout stays the hang guard.
            vs_pypy = None if backend == "wasm" else max_pypy_ratio
            self._run_backend_bench(
                backend, name, path, timeout,
                None, vs_pypy, t_cpython, t_pypy, pypy_output,
                max_rss_mb=None if backend == "wasm" else max_rss_mb,
            )

    def run_synthetic_suite(self):
        pattern = self.args.synthetic_pattern
        paths = sorted(Path(SYNTHETIC_BENCH_DIR).glob(pattern))
        if not paths and not Path(pattern).suffix:
            paths = sorted(Path(SYNTHETIC_BENCH_DIR).glob(f"{pattern}.py"))
        paths = [p for p in paths if p.is_file() and p.suffix == ".py"]
        if not paths:
            print(f"{red('ERROR')}: no synthetic benchmarks matched {pattern!r}")
            sys.exit(1)

        print(bold("synthetic parity suite"))
        print(dim(f"{len(paths)} benchmark(s), pattern={pattern!r}"))
        for path in paths:
            try:
                selfcheck = synth_selfcheck(path)
                skip_backends = synth_skip_backends(path) if selfcheck else ()
            except ValueError as e:
                print(f"{red('ERROR')}: {e}")
                sys.exit(1)
            if selfcheck:
                self.run_selfcheck(
                    f"synth/{path.stem}",
                    str(path),
                    self.args.synthetic_timeout,
                    skip_backends=skip_backends,
                )
            else:
                self.run_synthetic_bench(
                    str(path), self.args.synthetic_timeout,
                )
        # A fixture that loses its cpython reference also loses the
        # cpython/pypy output cross-check, so the count belongs in the summary
        # rather than only in the per-fixture line it scrolled past. A fixture
        # appearing here for the first time is a fixture whose baseline just
        # got weaker.
        if self.cpython_reference_skips:
            names = ", ".join(self.cpython_reference_skips)
            print(
                dim(
                    f"cpython reference skipped (>{SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S:g}s) "
                    f"for {len(self.cpython_reference_skips)}: {names}"
                )
            )
        # A wasm run with no usable dynasm denominator beside it leaves
        # WASM_MAX_DYNASM_RATIO with nothing to divide by. Said out loud
        # because the per-fixture line for an unevaluated gate is the same
        # green as a satisfied one.
        if self.wasm_ratio_ungated:
            print(
                dim(
                    f"wasm/dynasm {WASM_MAX_DYNASM_RATIO:g}x ratio not evaluated for "
                    f"{len(self.wasm_ratio_ungated)} fixture(s): dynasm did not run "
                    f"them in this invocation, or its execution-only time stayed "
                    f"under {FLOOR_GATE_MIN_BASELINE_S * 1000:g}ms"
                )
            )
        # Counters a fixture's header took out of its own gate. Named for the
        # same reason as the two above: the fixture prints green either way, and
        # the exemption is only reviewable if it is said out loud.
        if self.jitstats_ungated:
            print(
                dim(
                    f"jit-stats counters ungated by header for "
                    f"{len(self.jitstats_ungated)}: "
                    + "; ".join(self.jitstats_ungated)
                )
            )
        # The same weaker baseline, asked for rather than discovered. Listed
        # apart so the two are not read as one number: this one only grows when
        # someone writes the header line.
        if self.cpython_declared_skips:
            names = ", ".join(self.cpython_declared_skips)
            print(
                dim(
                    f"cpython reference skipped (header) "
                    f"for {len(self.cpython_declared_skips)}: {names}"
                )
            )

    # ── printing ──

    def print_backend_config(self):
        parts = []
        for b in ALL_BACKENDS:
            if self.enabled(b):
                label = b if b != "wasm" else f"wasm/{WASM_ENGINE}"
                parts.append(f"{label}={self._pyre(b)}(x{self._timeout_scale(b)})")
        if parts:
            print(f"backend: {' '.join(parts)}")

    def print_comparison_table(self):
        if not self.comparisons:
            return
        cols = [b for b in ALL_BACKENDS if self.enabled(b)]
        if not cols:
            return

        print(bold("Comparison"))

        header = f"  {'benchmark':<35s} {'cpython':>8s} {'pypy':>8s}"
        header += "".join(f" {b:>18s}" for b in cols)
        print(header)
        if any("~" in c[b] for c in self.comparisons for b in cols):
            print(
                "  ~ pypy exec clamped to floor; ratio is not a measurement, "
                "and no ratio gate is applied to it"
            )
        # Sniffing the rendered cell, like the line above: no other cell value
        # contains a `?` (they are times, ratios, FAIL/WRONG/UNSTABLE/skip/-).
        if any("?" in c[b] for c in self.comparisons for b in cols):
            print(
                "  ? pypy exec is above the floor but under "
                "FLOOR_GATE_MIN_BASELINE_S; the ceiling is applied, the floor "
                "gate declines the same baseline as too small to judge"
            )
        print("  " + "─" * (54 + 19 * len(cols)))
        for c in self.comparisons:
            row = f"  {c['name']:<35s} {c['cpython']:>8s} {c['pypy']:>8s}"
            row += "".join(f" {c[b]:>18s}" for b in cols)
            print(row)

    def print_summary(self):
        print()
        self.print_comparison_table()
        print()

        if self.wasm_ratio_allowed:
            names = ", ".join(
                f"{name} {ceiling:g}x"
                for name, ceiling in sorted(set(self.wasm_ratio_allowed))
            )
            print(
                dim(
                    f"wasm/dynasm ratio raised above {WASM_MAX_DYNASM_RATIO:g}x by "
                    f"`# pyre-check: max-wasm-ratio` for: {names}"
                )
            )

        if self.results:
            print("─" * 53)
            for r in self.results:
                print(f"  {r}")
            print("─" * 53)

        failed_runs = 0
        enabled_runs = 0
        for b in ALL_BACKENDS:
            if not self.enabled(b):
                continue
            enabled_runs += 1
            if self.fail_count.get(b, 0) > 0:
                failed_runs += 1

        # The jit-stats gate runs on every non-record run, so its tally is
        # reported on every non-record run — not only under --snapshot-diff,
        # where it used to live and where the bare CI invocation never looked.
        if self.args.snapshot_mode != "record":
            for paint, label, benches in (
                (red, "jit-stats regressed", self.jitstats_diffs),
                (yellow, "jit-stats improved (re-record)", self.jitstats_improvements),
                (yellow, "jit-stats unstable (not gated)", self.jitstats_unstable),
                (yellow, "jit-stats within band (not gated)", self.jitstats_banded),
                (red, "jit-stats baseline missing", self.jitstats_missing),
                (red, "jit-stats line absent", self.jitstats_absent),
                (red, "jit-stats census vacuous", self.jitstats_vacuous),
            ):
                if benches:
                    print(
                        f"{paint(label)}: {len(benches)} bench(es)"
                        f" — {' '.join(benches)}"
                    )

        if self.args.snapshot_mode or self.args.threshold is not None:
            if self.args.snapshot_mode == "record":
                print(dim(f"snapshot recorded under {SNAP_DIR}/"))
            elif self.args.snapshot_mode == "diff":
                if self.snapshot_diffs:
                    print(
                        f"{red('snapshot diff')}: {len(self.snapshot_diffs)} bench(es)"
                        f" — {' '.join(self.snapshot_diffs)}"
                    )
                if self.snapshot_missing:
                    print(
                        f"{dim('snapshot missing')}: {len(self.snapshot_missing)} bench(es)"
                        f" — {' '.join(self.snapshot_missing)}"
                    )
                if not self.snapshot_diffs and not self.snapshot_missing:
                    print(dim("snapshot diff: clean"))
            if self.args.threshold is not None:
                print(dim(f"threshold: ±{self.args.threshold}% vs baseline"))
            print()

        for b in ALL_BACKENDS:
            if not self.enabled(b):
                continue
            p = self.pass_count.get(b, 0)
            f = self.fail_count.get(b, 0)
            if f > 0:
                print(f"{red('FAILED')}: {b} {f} failed, {p} passed")
            else:
                print(f"{green('ALL PASSED')}: {b} {p}/{p}")

        if failed_runs > 0:
            print(f"{red('FAILED')}: {failed_runs} backend run(s) failed")
        else:
            print(f"{green('ALL PASSED')}: {enabled_runs}/{enabled_runs} backend run(s)")

        return 1 if failed_runs > 0 else 0


# ── Argument parsing ─────────────────────────────────────────────────

def parse_args():
    def positive_float(value):
        f = float(value)
        if f <= 0:
            raise argparse.ArgumentTypeError("must be greater than 0")
        return f

    def parse_backend_specs(specs):
        if specs is None:
            return list(DEFAULT_BACKENDS)

        backends = []
        for spec in specs:
            for backend in spec.split(","):
                backend = backend.strip()
                if not backend:
                    continue
                if backend not in CARGO_CONFIG:
                    choices = ", ".join(CARGO_CONFIG)
                    raise argparse.ArgumentTypeError(
                        f"invalid backend {backend!r}; choose from: {choices}"
                    )
                if backend not in backends:
                    backends.append(backend)

        if not backends:
            return list(DEFAULT_BACKENDS)
        return backends

    parser = argparse.ArgumentParser(
        description="pyre pre-merge check: correctness + regression guard + comparison",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--backend",
        action="append",
        nargs="?",
        const=",".join(DEFAULT_BACKENDS),
        default=None,
        metavar="BACKENDS",
        help="comma-separated backend list; may be repeated "
        f"(default: {','.join(DEFAULT_BACKENDS)})",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="use existing release backend binaries instead of running cargo build",
    )
    parser.add_argument(
        "--wasm-engine",
        choices=["wasmtime", "wasmi"],
        default="wasmtime",
        help="wasm runtime for the wasm backend: wasmtime (cranelift JIT, fast "
        "but recompiles the module each start) or wasmi (interpreter, near-zero "
        "startup, slower loops)",
    )
    parser.add_argument(
        "--no-startup-subtract",
        action="store_true",
        help="do not measure/subtract each interpreter's empty-program user-CPU "
        "startup from ratios and perf gates (report raw times incl. startup)",
    )
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument("--dynasm-timeout-scale", type=float, default=None)
    parser.add_argument("--cranelift-timeout-scale", type=float, default=None)
    parser.add_argument("--wasm-timeout-scale", type=float, default=None)
    parser.add_argument("--snapshot", dest="snapshot_mode", action="store_const", const="record")
    parser.add_argument("--snapshot-diff", dest="snapshot_mode", action="store_const", const="diff")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also run cpython on benchmarks without a vs_cpython gate (comparison only)",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="skip pyre/bench/synth feature-parity benchmarks",
    )
    parser.add_argument(
        "--cpython-suite",
        action="store_true",
        help="also run the vendored CPython suite gate (pyre/cpython_tests); "
             "off by default because it dominates this script's wall time",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="run only pyre/bench/synth feature-parity benchmarks",
    )
    parser.add_argument(
        "--synthetic-pattern",
        default="*.py",
        help="glob pattern under pyre/bench/synth for synthetic runs",
    )
    parser.add_argument(
        "--synthetic-timeout",
        type=positive_float,
        default=20.0,
        help="per-script timeout in seconds for synthetic benchmarks",
    )
    parser.add_argument("pyre_path", nargs="?", default="")
    args = parser.parse_args()
    try:
        args.backends = parse_backend_specs(args.backend)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    if args.pyre_path and args.backend is None:
        parser.error("[path/to/pyre] requires --backend when running a single binary")

    if args.pyre_path and len(args.backends) != 1:
        parser.error("[path/to/pyre] can only be used with exactly one --backend")

    if args.synthetic_only and args.no_synthetic:
        parser.error("--synthetic-only cannot be combined with --no-synthetic")

    if args.snapshot_mode == "record" and args.threshold is not None:
        print("NOTE: --threshold ignored in --snapshot record mode")
        args.threshold = None

    return args


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    global WASM_ENGINE
    WASM_ENGINE = args.wasm_engine
    chk = Check(args)

    backends = args.backends

    for backend in backends:
        if not args.no_build:
            chk.build_backend(backend)
        pyre_bin = args.pyre_path if args.pyre_path else default_binary(backend)
        if not Path(pyre_bin).exists():
            alt = pyre_bin + EXE
            if Path(alt).exists():
                pyre_bin = alt
        if not os.access(pyre_bin, os.X_OK) and not Path(pyre_bin).exists():
            if args.no_build:
                print(
                    f"ERROR: --no-build requested for backend '{backend}', but "
                    f"the executable is missing: {pyre_bin}"
                )
            else:
                print(
                    f"ERROR: build failed for backend '{backend}' "
                    f"(missing executable: {pyre_bin})"
                )
            sys.exit(1)
        if backend == "wasm" and args.no_build and not Path(WASM_MODULE_PATH).is_file():
            print(
                "ERROR: --no-build requested for backend 'wasm', but the "
                f"wasm-host module is missing: {WASM_MODULE_PATH}"
            )
            sys.exit(1)
        chk._set_pyre(backend, pyre_bin)

    print()
    print(bold("pyre pre-merge check"))
    chk.print_backend_config()
    print()
    chk.measure_startups()
    if not args.synthetic_only:
        chk.warmup(f"{BENCH_DIR}/int_loop.py")
        print()

        B = BENCH_DIR

        # fib_recursive / raise_catch / nbody / fannkuch are heavy on wasm
        # (interpreter round-trips + per-object Rust-heap allocation that the
        # wasm path does not yet collect, so they hit their timeout or the
        # linear-memory budget). They are intentionally NOT skipped on wasm:
        # we run them so the wasm leak/perf work can be driven down one bench
        # at a time, accepting the timeout state until each is fixed.

        # Gate values are set from the gate's OWN metric — user-CPU minus the
        # measured empty-program startup, min-of-5 interleaved — at ~2.5x the
        # measured ratio, so a runner 2.5x slower than an idle local box still
        # passes. A gate is only tightened when its baseline is healthy
        # (baseline exec >= ~5x that runtime's startup); where it is not, the
        # workload is sized up instead (see spectral_norm.py). Benches with
        # under ~3x headroom are deliberately left alone: nested_loop,
        # raise_catch and fib_recursive-vs-cpython are the known slow-runner
        # flakes. int_loop-cranelift recurrently measures 2.0-2.1x on the CI
        # runner (2x-tight), so its c_vs_py gate is raised 2->3.
        #             name              script                          timeout  d_vs_cp  d_vs_py  c_vs_cp  c_vs_py
        chk.run_bench("int_loop",       f"{B}/int_loop.py",             5,       None,    2,       None,    3)
        chk.run_bench("float_loop",     f"{B}/float_loop.py",           5,       None,    1.5,     None,    1.5)
        chk.run_bench("fib_loop",       f"{B}/fib_loop.py",             5,       2,       3,       2,       3)
        chk.run_bench("inline_helper",  f"{B}/inline_helper.py",        5,       None,    1.5,     None,    1.5)
        chk.run_bench("fib_recursive",  f"{B}/fib_recursive.py",        5,       2,       6,       2,       8)
        chk.run_bench("nested_loop",    f"{B}/nested_loop.py",          5,       None,    2,       None,    3)
        chk.run_bench("raise_catch",    f"{B}/raise_catch_loop.py",     5,       None,    1.5,     None,    2.5)
        # spectral_norm's hosts spread 0.5x-4.3x against pypy: pypy runs it in
        # 0.13s on the linux and macos runners and 0.39s on the windows one, so
        # pyre reads faster than pypy there. This is the bench that sets the
        # floor divisor's lower bound among the non-synthetic ones.
        chk.run_bench("spectral_norm",  f"{B}/spectral_norm.py",       15,       1,       5,       1,       5)
        chk.run_bench("nbody",          f"{B}/nbody.py",               10,       0.5,     5,       1,       5,    wasm_float_tol=True)
        chk.run_bench("fannkuch",       f"{B}/fannkuch.py",            30,       1,       5,       2,       15)
        # The branchy-inlined-callee guard (gh#343) lives in the synthetic parity
        # suite as bridge_branchy_callee.py, gated against pypy by
        # `# pyre-check: max-pypy-ratio`; a decline that keeps every crossing
        # interpreted (~34x) balloons the ratio past the limit.

    if not args.no_synthetic:
        print()
        chk.run_synthetic_suite()

    if args.cpython_suite and not args.synthetic_only:
        print()
        print(bold("vendored CPython suite"))
        chk.run_cpython_suite()

    rc = chk.print_summary()
    sys.exit(rc)


if __name__ == "__main__":
    main()
