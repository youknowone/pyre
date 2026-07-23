#!/usr/bin/env python3
"""pyre pre-merge check: correctness + regression guard + comparison

Cross-platform Python translation of pyre/check.sh.
"""

import argparse
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

# Opt-out (`--no-fbw-inline-multiframe`): export PYRE_FBW_INLINE_MULTIFRAME=0 into
# pyre child runs to exercise the #68 multi-frame inline rollback escape hatch.
# The path is on by default, so the default run already parity-checks it; this
# opt-out validates the flag-off fallback.
FBW_INLINE_MULTIFRAME_OFF = False

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
STARTUP_SAMPLES = 3
EXEC_TIME_FLOOR_S = WIN_TIMER_QUANTUM_S if sys.platform == "win32" else 0.005
# A single slow sample is retried before failing a performance gate. Windows
# needs more samples because its process CPU accounting is scheduler-tick
# quantized (see WIN_TIMER_QUANTUM_S above).
PERF_RETRY_RUNS = 5 if sys.platform == "win32" else 3
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

# ── Child-process user CPU time ──────────────────────────────────────

def _run_timed_unix(args, timeout_s, env=None):
    import resource
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
        proc.stdout.decode("utf-8", errors="replace"),
        utime,
        proc.returncode,
        proc.stderr.decode("utf-8", errors="replace"),
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
    """
    env = dict(os.environ)
    env["MAJIT_STRICT"] = "1"
    env["MAJIT_STATS"] = "1"
    # Pin the vendored, `_sre.MAGIC`-matched stdlib so pyre never picks up a
    # version-mismatched host `python3` off the PATH. An explicit PYRE_STDLIB
    # in the environment wins.
    if PYRE_STDLIB and "PYRE_STDLIB" not in env:
        env["PYRE_STDLIB"] = PYRE_STDLIB
    if FBW_INLINE_MULTIFRAME_OFF:
        env["PYRE_FBW_INLINE_MULTIFRAME"] = "0"
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
    """Return the last jit-stats line as normalized key/value text."""
    stats_line = None
    for line in stderr.splitlines():
        if line.startswith("[jit-stats]"):
            stats_line = line
    if stats_line is None:
        return None

    fields = {}
    for token in stats_line[len("[jit-stats]") :].split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    # This watches what the JIT compiles, never how well. A regression that
    # changes no structure (for example, an extra spill) is invisible here.
    return "".join(f"{key}={fields[key]}\n" for key in sorted(fields))


def _jit_stats_diff(saved, current, limit=6):
    """Describe normalized jit-stats changes, capped for terminal output."""
    def parse(snapshot):
        if snapshot is None:
            return {}
        return dict(line.split("=", 1) for line in snapshot.splitlines())

    old_fields = parse(saved)
    new_fields = parse(current)
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


# Sign-stable jit-stats counters: a nonzero value always means something went
# wrong (a trace aborted, or an internal compile bug fell back to the
# interpreter), never a tuning choice. They read 0 in every healthy baseline on
# every platform, so a rise above baseline is a regression the regression floor
# gates on unconditionally. The count-valued fields (guard_failures,
# loops_compiled, bridges_compiled) are deliberately excluded: their absolute
# value is not yet confirmed stable across runners, so they stay informational.
JITSTATS_BADNESS_FIELDS = ("loops_aborted", "internal_compile_panics")


def _jit_stats_regression_floor(saved, current):
    """Return a failure reason if any badness counter rose above its committed
    baseline, else "". A rise means the JIT started aborting traces or hitting
    internal compile panics it did not before — a defect on any platform. A
    counter that falls (an improvement) or holds never gates, and a field
    missing from either side is read as 0 so it cannot fire spuriously."""
    def parse(snapshot):
        if snapshot is None:
            return {}
        return dict(line.split("=", 1) for line in snapshot.splitlines())

    old_fields = parse(saved)
    new_fields = parse(current)
    regressions = []
    for field in JITSTATS_BADNESS_FIELDS:
        try:
            base = int(old_fields.get(field, 0))
            cur = int(new_fields.get(field, 0))
        except ValueError:
            continue
        if cur > base:
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
    return f"{t}s"


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
        # needed. Only the sign-stable badness counters gate here, so this stays
        # flake-free where the full-exact jit-stats diff (below, opt-in) cannot:
        # healthy baselines carry 0 for both on every platform, so a trace that
        # starts aborting or panicking is a real defect regardless of runner.
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
                if jitstats != saved_jitstats:
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

    def _run_backend_bench(
        self, backend, name, script, timeout,
        vs_cpython, vs_pypy, t_cpython, t_pypy, pypy_output,
        wasm_float_tol=False,
    ):
        pyre_bin = self._pyre(backend)
        effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))

        sys.stdout.write(f"    {backend:<10s}")
        sys.stdout.flush()

        output, elapsed, code, stderr = run_timed(
            [pyre_bin, script], timeout_s=effective_timeout, env=pyre_env(),
        )

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
            exp = pypy_output[:60]
            act = output[:60]
            self._record(backend, False, name, "wrong output")
            print(f"{red('WRONG')}  got: {act} expected(pypy): {exp}")
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
                self._record(
                    backend, False, name,
                    f"{checked_elapsed:.2f}s > cpython {checked_baseline:.2f}s x{vs_cpython}",
                )
                suffix = f" ({retry_note})" if retry_note else ""
                print(f"{red('SLOWER')}  pyre {checked_elapsed:.2f}s > cpython {checked_baseline:.2f}s x{vs_cpython}{suffix}")
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
                self._record(
                    backend, False, name,
                    f"{checked_elapsed:.2f}s > pypy {checked_baseline:.2f}s x{vs_pypy}",
                )
                suffix = f" ({retry_note})" if retry_note else ""
                print(f"{red('SLOWER')}  pyre {checked_elapsed:.2f}s > pypy {checked_baseline:.2f}s x{vs_pypy}{suffix}")
                self._append_comparison(
                    backend, name, t_cpython, t_pypy,
                    fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
                )
                return
            if retry_note:
                elapsed = checked_elapsed
                t_pypy = checked_baseline
                ratio = _ratio(elapsed, t_pypy)

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
        except ValueError as e:
            print(f"{red('ERROR')}: {e}")
            sys.exit(1)

        print(f"  {name}")

        sys.stdout.write(f"    {'cpython':<10s}")
        sys.stdout.flush()
        cpython_output, cpython_time, cpython_code, _ = run_timed(
            [PYTHON3, path], timeout_s=effective_timeout,
        )
        if cpython_code != 0:
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
            # wasm carries no perf-ratio gate, matching `run_bench` (which has no
            # `wasm_vs_*` parameter at all): it legitimately runs a few× slower
            # than the native backends a fixture's ratio is tuned for, so one
            # ratio cannot gate both — see WASM_TIMEOUT_SCALE. Its per-bench
            # timeout stays the hang guard.
            vs_pypy = None if backend == "wasm" else max_pypy_ratio
            self._run_backend_bench(
                backend, name, path, timeout,
                None, vs_pypy, t_cpython, t_pypy, pypy_output,
            )

    def run_synthetic_suite(self):
        pattern = self.args.synthetic_pattern
        paths = sorted(Path(SYNTHETIC_BENCH_DIR).glob(pattern))
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
    parser.add_argument(
        "--no-fbw-inline-multiframe",
        action="store_true",
        help="run pyre with PYRE_FBW_INLINE_MULTIFRAME=0 (#68 forward-branch "
        "multi-frame inline is on by default; this exercises the rollback "
        "escape hatch)",
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
    if args.no_fbw_inline_multiframe:
        global FBW_INLINE_MULTIFRAME_OFF
        FBW_INLINE_MULTIFRAME_OFF = True
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

        #             name              script                          timeout  d_vs_cp  d_vs_py  c_vs_cp  c_vs_py
        chk.run_bench("int_loop",       f"{B}/int_loop.py",             5,       None,    2,       None,    2)
        chk.run_bench("float_loop",     f"{B}/float_loop.py",           5,       None,    1.5,     None,    1.5)
        chk.run_bench("fib_loop",       f"{B}/fib_loop.py",             5,       2,       3,       2,       3)
        chk.run_bench("inline_helper",  f"{B}/inline_helper.py",        5,       None,    1.5,     None,    1.5)
        chk.run_bench("fib_recursive",  f"{B}/fib_recursive.py",        5,       2,       13,      2,       13)
        chk.run_bench("nested_loop",    f"{B}/nested_loop.py",          5,       None,    2,       None,    3)
        chk.run_bench("raise_catch",    f"{B}/raise_catch_loop.py",     5,       None,    1.5,     None,    2.5)
        chk.run_bench("spectral_norm",  f"{B}/spectral_norm.py",        5,       2,       7,       2,       7)
        chk.run_bench("nbody",          f"{B}/nbody.py",               10,       1,       5,       1,       5,    wasm_float_tol=True)
        chk.run_bench("fannkuch",       f"{B}/fannkuch.py",            30,       1,       5,       2,       None)
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

    if not args.no_synthetic:
        print()
        chk.run_synthetic_suite()

    rc = chk.print_summary()
    sys.exit(rc)


if __name__ == "__main__":
    main()
