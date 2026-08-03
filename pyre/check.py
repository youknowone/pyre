#!/usr/bin/env python3
"""pyre pre-merge check: correctness + regression guard + comparison

Cross-platform Python translation of pyre/check.sh.
"""

import argparse
import difflib
import math
import os
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
# vendored `lib-python/3` are coupled to that version. Prefer a version-matched
# oracle so a stale `python3` on PATH (an older system CPython) does not diverge
# from pypy on version-sensitive error text and trip a spurious cpython-vs-pypy
# baseline mismatch.
PYTHON3 = os.environ.get("PYRE_CHECK_PYTHON3") or next(
    (cand for cand in ("python3.14", "python3", "python") if shutil.which(cand)),
    "python3",
)
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
# Native Windows CI can spend substantially more wall time than reported
# process user-CPU while antivirus and concurrent matrix jobs contend for the
# runner.  Keep the timeout as a hang guard by granting native backends 2x
# headroom; performance still gates on measured user-CPU below.
WIN_NATIVE_TIMEOUT_SCALE = 2.0

BENCH_DIR = "pyre/bench"
SYNTHETIC_BENCH_DIR = "pyre/bench/synth"
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
    env = dict(os.environ)
    env["MAJIT_STRICT"] = "1"
    env["MAJIT_STATS"] = "1"
    env["PYRE_DESCR_SPELLING_GATE"] = "1"
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

def _jit_stats_snapshot(stderr):
    """Return every jit-stats line merged into normalized key/value text.

    The interpreter emits more than one `[jit-stats]` line — the counters, the
    mc_diag tallies, the descr-set universe. Keeping only the last one dropped
    `loops_aborted` and `internal_compile_panics` from the snapshot, which
    silently disarmed `_jit_stats_regression_floor`: it read both counters as
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
    if not seen:
        return None
    fields = {k: v for k, v in fields.items() if k in JITSTATS_SNAPSHOT_FIELDS}
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


def _jit_stats_fields_equal(saved, current):
    """Whether two snapshots agree on every `JITSTATS_SNAPSHOT_FIELDS` key,
    reading a field missing from either side as "0"."""
    old_fields = _parse_jit_stats(saved)
    new_fields = _parse_jit_stats(current)
    return all(
        old_fields.get(field, "0") == new_fields.get(field, "0")
        for field in JITSTATS_SNAPSHOT_FIELDS
    )


def _jit_stats_diff(saved, current, limit=6):
    """Describe normalized jit-stats changes, capped for terminal output."""
    old_fields = _parse_jit_stats(saved)
    new_fields = _parse_jit_stats(current)
    changes = [
        f"{key} {old_fields.get(key, '<missing>')} -> "
        f"{new_fields.get(key, '<missing>')}"
        for key in sorted(old_fields.keys() | new_fields.keys())
        if old_fields.get(key) != new_fields.get(key)
    ]
    shown = changes[:limit]
    if len(changes) > limit:
        shown.append(f"... {len(changes) - limit} more")
    return ", ".join(shown)


# Direction-stable jit-stats counters: the regression floor gates on these two
# because a rise is a defect whatever the absolute value happens to be.
# `internal_compile_panics` counts an internal compile bug falling back to the
# interpreter, so 0 is its only healthy value. `loops_aborted` is not absolute:
# an abort is either a give-up the tracer is supposed to perform (an over-long
# trace hits `blackhole_if_trace_too_long`, pyjitpl.py:2865) or a walker
# coverage gap that is tracked elsewhere — the synth fixtures below carry
# `LoopBearingCalleeInlineUnsupported`, the walker's open gap on inlining a
# loop-bearing callee. Either way the baseline pins the count those known
# declines produce today and the gate fires on a rise, which means a loop the
# tracer used to finish has stopped finishing. `bridges_compiled` stays out
# entirely: it moves in both directions under ordinary tuning and neither
# direction is a defect on its own. The other two count-valued counters each
# have ONE defect direction and gate through their own tuple below —
# `guard_failures` on a bounded rise, `loops_compiled` on any fall.
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
JITSTATS_BADNESS_FIELDS = (
    "loops_aborted",
    "internal_compile_panics",
    "descr_set_absent",
    "descr_set_ambiguous",
    "descr_set_stale_absent",
)

# Counters gated on a bounded RISE rather than on any rise at all: a fall is
# free, and a rise fails only past `base + max(base // 4, 2)`.
#
# `guard_failures` counts every guard failure that re-enters the metainterp, so
# it is the counter that moves when a compiled loop's guards start failing more
# often — a cost `max-pypy-ratio` cannot see, because that ceiling is one
# absolute number per fixture: recursion_memo_branch sits near 8.9x under a
# limit of 20, so it absorbs a 2x regression and still passes. Narrowing
# `nested_break_bridge_resume_hazard` so that fixture's `main` compiles moves
# the counter 1757 -> 2613 (+49%) and measures slower, and no other gate here
# notices: nothing aborts, so the badness floor holds, and the ratio keeps 2.2x
# of slack.
#
# It is deterministic by construction, not by luck: a guard's counter hash comes
# from `JitCounter.fetch_next_hash` (counter.py:142-150), a pure sequence with
# no address or clock input, so the same program yields the same count. Measured
# bit-stable run-to-run on one host. The +25% band and the +2 absolute allowance
# are what cover cross-runner drift, which is NOT yet measured — they are sized
# to stay well under the class above, so a real storm still fires. The absolute
# term is for a fixture whose baseline is a handful of guards, where one more
# guard is not yet a storm.
JITSTATS_RISE_BOUNDED_FIELDS = ("guard_failures",)

# Counters gated on any FALL below the baseline. `loops_compiled` closes the
# blind spot the two tuples above leave open: a change that makes the tracer
# stop admitting a frame at all — a widened `unsupported_jit_shape` predicate, a
# pre-trace decline — aborts nothing (`loops_aborted` holds) and *lowers*
# `guard_failures` (the compiled guards that used to fail are gone), so both
# gates above read the loss as an improvement and the fixture goes green while
# its hot loop runs interpreted.
#
# A fall is gated exactly, with no band, because this is the most stable counter
# on the surface: across the 341 synthetic fixtures the dynasm and cranelift
# baselines — two independent code generators — agree on `loops_compiled` for
# 340 of them (99.7%), against 97.4% for `guard_failures`, which is why that one
# needs a band and this one does not.
#
# A fall is not *always* a defect: two loops merging into one trace lowers the
# count while raising coverage. That is the same bargain `loops_aborted` already
# makes in the other direction — an intentional structural change re-records its
# baseline, and the gate is what forces someone to look.
JITSTATS_FALL_FIELDS = ("loops_compiled",)

# What a committed `.jitstats` baseline is allowed to contain. `_jit_stats_snapshot`
# merges every `[jit-stats]` line so the floor above cannot be disarmed by
# reshuffling them, but the merged set also carries keys that are not comparable
# against a file checked into the repo:
#
# * `all_descrs` and `descr_set_resolved` are HOST-DEPENDENT — the Charon/LLBC
#   extraction runs per host, so the same commit reads `all_descrs` 1168 on
#   macOS, 1396 on ubuntu and 1365 on windows. Committing either makes every
#   host but one disagree.
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
    "guard_failures",
)


def _jit_stats_regression_floor(saved, current):
    """Return a failure reason if a gated counter rose above what its committed
    baseline allows, else "". For JITSTATS_BADNESS_FIELDS the allowance is the
    baseline itself: a rise means the JIT started aborting traces or hitting
    internal compile panics it did not before — a defect on any platform.
    JITSTATS_RISE_BOUNDED_FIELDS carries the band documented there, and
    JITSTATS_FALL_FIELDS gates the opposite direction. A counter that moves the
    way its group calls an improvement, or holds, never gates.

    A field missing from either side is read as 0 for the badness counters,
    whose healthy value IS 0, so they gate from the first run without
    re-recording. The count-valued counters have no such healthy value, so a
    baseline that does not name one is treated as not pinning it at all — a
    baseline recorded before the field existed must not read as `0 -> 1757`."""
    old_fields = _parse_jit_stats(saved)
    new_fields = _parse_jit_stats(current)
    regressions = []
    for field in (
        JITSTATS_BADNESS_FIELDS + JITSTATS_RISE_BOUNDED_FIELDS + JITSTATS_FALL_FIELDS
    ):
        if field not in JITSTATS_BADNESS_FIELDS and field not in old_fields:
            continue
        try:
            base = int(old_fields.get(field, 0))
            cur = int(new_fields.get(field, 0))
        except ValueError:
            continue
        if field in JITSTATS_RISE_BOUNDED_FIELDS:
            regressed = cur > base + max(base // 4, 2)
        elif field in JITSTATS_FALL_FIELDS:
            regressed = cur < base
        else:
            regressed = cur > base
        if regressed:
            regressions.append(f"{field} {base} -> {cur}")
    if regressions:
        return "jit-stats regression: " + ", ".join(regressions)
    return ""


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
    """
    prefix = "# pyre-check: max-pypy-ratio="
    with open(path, encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if not line.startswith(prefix):
                continue
            try:
                ratio = float(line[len(prefix):].strip())
            except ValueError as e:
                raise ValueError(f"invalid synthetic performance gate in {path}: {line.strip()}") from e
            if ratio <= 0:
                raise ValueError(f"synthetic performance gate must be positive in {path}")
            return ratio
    return None


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
        # Per-backend bookkeeping, keyed by backend name so any backend
        # (dynasm / cranelift / wasm / a single `--pyre-path` binary) is tracked
        # uniformly.
        self.pass_count = {}
        self.fail_count = {}
        self.pyre = {}
        # interpreter/backend key -> measured empty-program user-CPU startup (s).
        # Populated by measure_startups(); missing key => 0.0 (no subtraction).
        self.startup = {}
        self.snapshot_diffs = []
        self.snapshot_missing = []
        self.jitstats_diffs = []
        self.jitstats_missing = []

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

    def _jitstats_baseline_path(self, backend, script):
        # The committed structural-stats baseline sits beside its benchmark
        # source (pyre/bench/<name>.<backend>.jitstats), not in the gitignored
        # check.snap scratch tree that holds the local .out/.time snapshots.
        source = Path(script)
        return source.with_name(f"{source.stem}.{backend}.jitstats")

    def _apply_snapshot_gate(self, backend, name, script, output, stderr, elapsed):
        status, reason = "ok", ""
        out_path = self._snapshot_path(backend, name, "out")
        time_path = self._snapshot_path(backend, name, "time")
        jitstats_path = self._jitstats_baseline_path(backend, script)
        jitstats = _jit_stats_snapshot(stderr)

        # Regression floor — enforced on EVERY run, so a structural JIT
        # regression reddens the default `pyre/check.py` (locally, and in the
        # bare CI invocation) the instant it lands, with no --snapshot-diff
        # needed. Only the direction-stable counters gate here, so this stays
        # flake-free where the full-exact jit-stats diff (below, opt-in) cannot:
        # the baseline pins whatever aborts a bench legitimately records, and a
        # rise above it is a real defect regardless of runner.
        # This catches the inline-abort class (e.g. an inlined-callee LOAD_GLOBAL
        # fold that stops resolving: loops_aborted 0 -> 5). Record mode is
        # writing a fresh baseline, so it is exempt.
        if (
            self.args.snapshot_mode != "record"
            and jitstats is not None
            and jitstats_path.exists()
        ):
            floor = _jit_stats_regression_floor(
                jitstats_path.read_text(encoding="utf-8"), jitstats
            )
            if floor:
                return "fail", floor

        # A benchmark with no committed baseline runs every gate above against
        # nothing and prints PASS, so "never recorded" is indistinguishable from
        # "recorded and clean" — which is how the floor came to cover 3 of 340
        # synthetic fixtures without anyone noticing. Every benchmark that
        # reaches here emitted a `[jit-stats]` line, so the baseline is always
        # recordable; requiring it is what stops a newly added fixture from
        # silently reopening that hole.
        if (
            self.args.snapshot_mode != "record"
            and jitstats is not None
            and not jitstats_path.exists()
        ):
            return "fail", (
                f"no committed jit-stats baseline ({jitstats_path.name}) — record it with "
                f"`pyre/check.py --snapshot --backend {backend}`"
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

            if not jitstats_path.exists():
                self.jitstats_missing.append(f"{backend}/{name}")
            else:
                saved_jitstats = jitstats_path.read_text(encoding="utf-8")
                # Compare key-wise with a missing field read as "0", the same
                # semantics `_jit_stats_regression_floor` documents. A baseline
                # recorded before a counter existed is then still equal to a
                # run that reports it as 0, so adding an invariant counter
                # costs no re-record — only a real change diffs.
                if not _jit_stats_fields_equal(saved_jitstats, jitstats):
                    self.jitstats_diffs.append(f"{backend}/{name}")
                    delta = _jit_stats_diff(saved_jitstats, jitstats)
                    return "fail", f"jit-stats diff: {delta}"

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
        # mtime and needlessly invalidate the runner's `<module>.cwasm` compiled
        # cache (which is keyed by mtime), forcing a ~5s recompile on every run.
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
        """
        runner = default_binary("wasm")
        if not Path(runner).exists():
            return
        env = dict(os.environ)
        env["PYRE_WASM_MODULE"] = str(Path(WASM_MODULE_PATH).resolve())
        env["PYRE_WASM_ENGINE"] = "wasmtime"
        print("Warming wasmtime module cache (.cwasm)...")
        try:
            subprocess.run(
                [runner, "--engine", "wasmtime", os.devnull],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, timeout=120,
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
    ):
        """Return median (pyre, baseline) timings after a slow first sample.

        A retry is deliberately limited to the failed gate: correctness has
        already passed, and a nonzero exit, changed output, or JIT panic in any
        retry remains a failure rather than being hidden by a median.
        """
        attempts = PERF_RETRY_RUNS
        effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))
        pyre_times = []
        baseline_times = []
        for _ in range(attempts):
            _, baseline_time, baseline_code, _ = run_timed(
                baseline_cmd, timeout_s=effective_timeout,
            )
            if baseline_code != 0:
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

    def _performance_gate_passed(
        self, backend, script, timeout, elapsed, limit, baseline_time,
        baseline_cmd, expected_output, baseline_key,
    ):
        """Check one performance ratio, retrying a failure by median.

        The pass/fail decision compares execution-only (startup-subtracted)
        times; the returned elapsed/baseline stay raw for reporting.
        """
        # Each execution-only value is a difference between two independently
        # measured quantities (bench - empty-program startup).  On Windows each
        # quantity may be off by one scheduler tick, so the comparison
        # ``pyre <= baseline * limit`` needs 2q on the left and 2q*limit on the
        # right.  Adding one tick to every raw sample cannot model this: those
        # additions cancel during startup subtraction.
        compare_buffer = BENCH_COMPARE_BUFFER_S
        if sys.platform == "win32":
            compare_buffer += 2 * WIN_TIMER_QUANTUM_S * (1 + limit)

        if (
            self._exec_time(backend, elapsed)
            <= self._exec_time(baseline_key, baseline_time) * limit
            + compare_buffer
        ):
            return True, elapsed, baseline_time, ""

        retry = self._retry_performance_gate(
            backend, script, timeout, baseline_cmd, expected_output,
        )
        if retry is not None:
            median_elapsed, median_baseline = retry
            if (
                self._exec_time(backend, median_elapsed)
                <= self._exec_time(baseline_key, median_baseline) * limit
                + compare_buffer
            ):
                return True, median_elapsed, median_baseline, f"median {PERF_RETRY_RUNS}"
            return False, median_elapsed, median_baseline, f"median {PERF_RETRY_RUNS}"

        return False, elapsed, baseline_time, ""

    def _gate_fail_detail(self, backend, baseline, measured, baseline_time, limit):
        """One-line FAIL detail using the exact numbers the gate compared.

        The gate decides on startup-subtracted exec times
        (``_exec_time(backend, measured) <= _exec_time(baseline, baseline_time)
        * limit``), so those exec times — not the raw run times — are printed,
        alongside their true measured ratio and the gate threshold. On a FAIL
        the ratio necessarily exceeds the threshold, so every number on the
        line is arithmetically self-consistent: exec_measured / exec_baseline
        equals the shown ratio, which is above the shown gate.
        """
        exec_m = self._exec_time(backend, measured)
        exec_b = self._exec_time(baseline, baseline_time)
        if exec_b in (None, "-") or float(exec_b) <= 0:
            ratio = "-"
        else:
            ratio = f"{float(exec_m) / float(exec_b):.1f}x"
        # A baseline sitting exactly on EXEC_TIME_FLOOR_S was clamped: the
        # subtraction put it at or below the floor, so its execution time could
        # not be separated from its own startup and the ratio computed against
        # it is an artifact of the clamp rather than a measurement.  Say so on
        # the line, because the printed denominator otherwise reads like one.
        clamped = (
            not self.args.no_startup_subtract
            and exec_b not in (None, "-")
            and float(exec_b) <= EXEC_TIME_FLOOR_S
        )
        detail = (
            f"exec {exec_m:.2f}s > {baseline} {exec_b:.2f}s  "
            f"ratio {ratio} > gate {float(limit):g}x"
        )
        if clamped:
            detail += f"  [{baseline} exec clamped to floor; ratio not a measurement]"
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

        def _ratio(elapsed_val, pypy_val):
            num = self._exec_time(backend, elapsed_val)
            den = self._exec_time("pypy", pypy_val)
            if den in (None, "-") or float(den) <= 0 or num in (None, "-"):
                return "-"
            return f"{float(num) / float(den):.1f}x"

        ratio = _ratio(elapsed, t_pypy) if elapsed > 0 else "-"

        retry_note = ""
        if vs_cpython and t_cpython not in (None, "-"):
            passed, checked_elapsed, checked_baseline, retry_note = self._performance_gate_passed(
                backend, script, timeout, elapsed, vs_cpython, float(t_cpython),
                [PYTHON3, script], pypy_output, "cpython",
            )
            if not passed:
                detail = self._gate_fail_detail(
                    backend, "cpython", checked_elapsed, checked_baseline, vs_cpython,
                )
                self._record(backend, False, name, detail)
                suffix = f" ({retry_note})" if retry_note else ""
                print(f"{red('SLOWER')}  pyre {detail}{suffix}")
                self._append_comparison(
                    backend, name, t_cpython, t_pypy,
                    fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
                )
                return
            if retry_note:
                elapsed = checked_elapsed
                ratio = _ratio(elapsed, t_pypy)

        if vs_pypy and t_pypy not in (None, "-"):
            passed, checked_elapsed, checked_baseline, retry_note = self._performance_gate_passed(
                backend, script, timeout, elapsed, vs_pypy, float(t_pypy),
                [PYPY3, script], pypy_output, "pypy",
            )
            if not passed:
                detail = self._gate_fail_detail(
                    backend, "pypy", checked_elapsed, checked_baseline, vs_pypy,
                )
                self._record(backend, False, name, detail)
                suffix = f" ({retry_note})" if retry_note else ""
                print(f"{red('SLOWER')}  pyre {detail}{suffix}")
                self._append_comparison(
                    backend, name, t_cpython, t_pypy,
                    fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
                )
                return
            if retry_note:
                elapsed = checked_elapsed
                t_pypy = checked_baseline
                ratio = _ratio(elapsed, t_pypy)

        # `# pyre-check: max-rss-mb=` — the axis the ratio gates cannot see.
        # Checked after them so a fixture that is both slow and fat reports the
        # slowness first, matching the existing gate order.
        if max_rss_mb is not None and peak_rss_mb is not None and peak_rss_mb > max_rss_mb:
            detail = f"peak RSS {peak_rss_mb:.0f}MB > {max_rss_mb:.0f}MB"
            self._record(backend, False, name, detail)
            print(f"{red('FATTER')}  pyre {detail}")
            self._append_comparison(
                backend, name, t_cpython, t_pypy,
                fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
            )
            return

        snap_status, snap_reason = self._apply_snapshot_gate(
            backend, name, script, output, stderr, elapsed,
        )
        if snap_status == "fail":
            self._record(backend, False, name, snap_reason)
            print(f"{red('SNAPDIFF')}  {snap_reason}")
            self._append_comparison(backend, name, t_cpython, t_pypy, "SNAPDIFF")
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
        skip_backends=(), wasm_float_tol=False,
    ):
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
        guards whose signal is a timing ratio, not byte-identical output, so
        they cannot go through `run_bench` or the synthetic suite.

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

    # ── synthetic parity suite ──

    def run_synthetic_bench(self, path, timeout):
        name = f"synth/{Path(path).stem}"
        effective_timeout = scaled_timeout(timeout, self.args.timeout_scale)
        try:
            max_pypy_ratio = synth_perf_gate(path)
            max_rss_mb = synth_rss_gate(path)
            skip_backends = synth_skip_backends(path)
        except ValueError as e:
            print(f"{red('ERROR')}: {e}")
            sys.exit(1)

        print(f"  {name}")

        sys.stdout.write(f"    {'cpython':<10s}")
        sys.stdout.flush()
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
            cpython_output = None
            t_cpython = "-"
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
        print("  " + "─" * (54 + 19 * len(cols)))
        for c in self.comparisons:
            row = f"  {c['name']:<35s} {c['cpython']:>8s} {c['pypy']:>8s}"
            row += "".join(f" {c[b]:>18s}" for b in cols)
            print(row)

    def print_summary(self):
        print()
        self.print_comparison_table()
        print()

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
                if self.jitstats_diffs:
                    print(
                        f"{red('jit-stats diff')}: {len(self.jitstats_diffs)} bench(es)"
                        f" — {' '.join(self.jitstats_diffs)}"
                    )
                if self.jitstats_missing:
                    print(
                        f"{dim('jit-stats missing')}: {len(self.jitstats_missing)} bench(es)"
                        f" — {' '.join(self.jitstats_missing)}"
                    )
                if not self.jitstats_diffs and not self.jitstats_missing:
                    print(dim("jit-stats diff: clean"))
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
        chk.build_backend(backend)
        pyre_bin = args.pyre_path if args.pyre_path else default_binary(backend)
        if not Path(pyre_bin).exists():
            alt = pyre_bin + EXE
            if Path(alt).exists():
                pyre_bin = alt
        if not os.access(pyre_bin, os.X_OK) and not Path(pyre_bin).exists():
            print(f"ERROR: build failed for backend '{backend}' (missing executable: {pyre_bin})")
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
        chk.run_bench("spectral_norm",  f"{B}/spectral_norm.py",       15,       1,       5,       1,       5)
        chk.run_bench("nbody",          f"{B}/nbody.py",               10,       0.5,     5,       1,       5,    wasm_float_tol=True)
        chk.run_bench("fannkuch",       f"{B}/fannkuch.py",            30,       1,       5,       2,       15)
        # Skipped on wasm: the guard times calls with `time.perf_counter()`, and
        # the wasm guest has no `time` module (import fails before any output),
        # so the guard is native-JIT-backend only.
        chk.run_selfcheck(
            "loop_reentry",
            f"{B}/loop_reentry_regression.py",
            15,
            skip_backends=("wasm",),
        )
        # Skipped on wasm: the guest has no os/filesystem — open() raises
        # NotImplementedError and `import os` has no posix backend — so the
        # errno-specific OSError subclass behaviour is native-JIT-backend only.
        chk.run_selfcheck(
            "oserror_errno_fields",
            f"{B}/oserror_errno_fields_regression.py",
            15,
            skip_backends=("wasm",),
        )
        # Skipped on wasm for the same reason: os.replace needs a filesystem.
        chk.run_selfcheck(
            "posix_replace",
            f"{B}/posix_replace_regression.py",
            15,
            skip_backends=("wasm",),
        )
        # The f_locals write-through it drives is a 3.14 FrameLocalsProxy
        # behaviour PyPy 3.11 lacks (its f_locals is a snapshot), so cpython and
        # pypy disagree on the mutated value and it cannot be a synthetic bench.
        # The guard is pyre-internal: a forcing callee's escape flush must resume
        # past the abort, else the in-flight FOR_ITER iteration drops and the
        # loop total comes up short. Asserted inside the script.
        chk.run_selfcheck(
            "getframe_escape_flush_writethrough",
            f"{B}/getframe_escape_flush_writethrough_regression.py",
            15,
        )
        # The coordinate a frame reports WHILE it is still running: compiled code
        # runs no per-opcode `last_instr` store, so a replayed frame answers for
        # the instruction it is on only if the blackhole publishes at the
        # `-live-` marker. Runs on every backend -- it is also the guard that
        # catches a store whose width overruns `valuestackdepth` onto the
        # `last_instr` next to it, which is a 32-bit-only failure.
        chk.run_selfcheck(
            "frame_lineno_mid_replay",
            f"{B}/frame_lineno_mid_replay_regression.py",
            20,
        )
        # The sibling shape: the frame handed out is the INLINED CALLEE's own,
        # not the virtualizable its caller runs as. That frame carries the `-1`
        # `last_instr` sentinel and is not what the escape flush writes, so its
        # `f_lineno` / `f_locals` are only right because `sys._getframe` forces
        # and the resulting abort finishes the iteration interpreted. Pins the
        # answer so a change that retires that abort has to keep it -- the whole
        # synthetic corpus stayed green while these three reads broke.
        chk.run_selfcheck(
            "frame_inlined_callee_own_image",
            f"{B}/frame_inlined_callee_own_image_regression.py",
            20,
        )
        # The branchy-inlined-callee guard (gh#343) lives in the synthetic parity
        # suite as bridge_branchy_callee.py, gated against pypy by
        # `# pyre-check: max-pypy-ratio`; a decline that keeps every crossing
        # interpreted (~34x) balloons the ratio past the limit.

    if not args.no_synthetic:
        print()
        chk.run_synthetic_suite()

    rc = chk.print_summary()
    sys.exit(rc)


if __name__ == "__main__":
    main()
