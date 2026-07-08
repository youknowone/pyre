#!/usr/bin/env python3
"""pyre pre-merge check: correctness + regression guard + comparison

Cross-platform Python translation of pyre/check.sh.
"""

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EXE = ".exe" if sys.platform == "win32" else ""
PYTHON3 = os.environ.get("PYRE_CHECK_PYTHON3") or (
    "python3" if shutil.which("python3") else "python"
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

BENCH_DIR = "pyre/bench"
SYNTHETIC_BENCH_DIR = "pyre/bench/synth"
SNAP_DIR = "pyre/check.snap"
BENCH_COMPARE_BUFFER_S = 0.005
# Windows `GetProcessTimes` / JobObject user-CPU accounting is quantized to
# the system scheduler tick (default 1/64 s ≈ 15.625 ms).  Any measured time
# could be off by up to one tick, so add one tick to every Windows
# measurement to absorb the quantization error.
WIN_TIMER_QUANTUM_S = 1.0 / 64

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
    utime += WIN_TIMER_QUANTUM_S
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

def scaled_timeout(base, scale):
    v = base * scale
    return int(v) if v == int(v) else float(f"{v:.3f}".rstrip("0").rstrip("."))


def fmt_time(t):
    if t is None or t == "-":
        return "-"
    return f"{t}s"


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
        self.snapshot_diffs = []
        self.snapshot_missing = []

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
        return self.args.timeout_scale

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

    def _apply_snapshot_gate(self, backend, name, output, elapsed):
        status, reason = "ok", ""
        out_path = self._snapshot_path(backend, name, "out")
        time_path = self._snapshot_path(backend, name, "time")

        if self.args.snapshot_mode == "record":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output, encoding="utf-8")
            time_path.write_text(f"{elapsed:.2f}", encoding="utf-8")

        if self.args.snapshot_mode == "diff":
            if not out_path.exists():
                self.snapshot_missing.append(f"{backend}/{name}")
            else:
                saved_out = out_path.read_text(encoding="utf-8")
                if output != saved_out:
                    self.snapshot_diffs.append(f"{backend}/{name}")
                    return "fail", "snapshot output diff"

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
                print("    scripts/extract-llbc.py pyre-object pyre-interpreter")
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

        ratio = "-"
        if t_pypy not in (None, "-") and float(t_pypy) > 0 and elapsed > 0:
            ratio = f"{elapsed / float(t_pypy):.1f}x"

        if vs_cpython and t_cpython not in (None, "-"):
            if elapsed > float(t_cpython) * vs_cpython + BENCH_COMPARE_BUFFER_S:
                self._record(
                    backend, False, name,
                    f"{elapsed:.2f}s > cpython {t_cpython}s x{vs_cpython}",
                )
                print(f"{red('SLOWER')}  pyre {elapsed:.2f}s > cpython {t_cpython}s x{vs_cpython}")
                self._append_comparison(
                    backend, name, t_cpython, t_pypy,
                    fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
                )
                return

        if vs_pypy and t_pypy not in (None, "-"):
            if elapsed > float(t_pypy) * vs_pypy + BENCH_COMPARE_BUFFER_S:
                self._record(
                    backend, False, name,
                    f"{elapsed:.2f}s > pypy {t_pypy}s x{vs_pypy}",
                )
                print(f"{red('SLOWER')}  pyre {elapsed:.2f}s > pypy {t_pypy}s x{vs_pypy}")
                self._append_comparison(
                    backend, name, t_cpython, t_pypy,
                    fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
                )
                return

        snap_status, snap_reason = self._apply_snapshot_gate(
            backend, name, output, elapsed,
        )
        if snap_status == "fail":
            self._record(backend, False, name, snap_reason)
            print(f"{red('SNAPDIFF')}  {snap_reason}")
            self._append_comparison(backend, name, t_cpython, t_pypy, "SNAPDIFF")
            return

        self._record(backend, True, name, f"{elapsed:.2f}s")
        print(f"{green('PASS')}  {elapsed:.2f}s")
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
            t_cpython = f"{t_cpu:.2f}"
            if cpython_code != 0:
                print(f"{red('CRASH')} (exit {cpython_code})")
            else:
                print(f"{dim('done')}  {t_cpython}s")

        sys.stdout.write(f"    {'pypy':<10s}")
        sys.stdout.flush()
        pypy_output, pypy_cpu, pypy_code, _ = run_timed([PYPY3, script])
        t_pypy = f"{pypy_cpu:.2f}" if pypy_code == 0 else "-"
        if pypy_code != 0:
            print(f"{red('CRASH')} (exit {pypy_code})")
            for backend in ALL_BACKENDS:
                if self.enabled(backend):
                    self._record(backend, False, name, "pypy crash")
                    self._append_comparison(backend, name, t_cpython, "-", "FAIL")
            return
        print(f"{dim('done')}  {t_pypy}s")

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

    # ── synthetic parity suite ──

    def run_synthetic_bench(self, path, timeout):
        name = f"synth/{Path(path).stem}"
        effective_timeout = scaled_timeout(timeout, self.args.timeout_scale)

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
            t_cpython = f"{cpython_time:.2f}"
            print(f"{dim('done')}  {t_cpython}s")

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
        t_pypy = f"{pypy_time:.2f}"
        print(f"{dim('done')}  {t_pypy}s")

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
            self._run_backend_bench(
                backend, name, path, timeout,
                None, None, t_cpython, t_pypy, pypy_output,
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

        self.print_comparison_table()
        print()

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
        "--wasm-engine",
        choices=["wasmtime", "wasmi"],
        default="wasmtime",
        help="wasm runtime for the wasm backend: wasmtime (cranelift JIT, fast "
        "but recompiles the module each start) or wasmi (interpreter, near-zero "
        "startup, slower loops)",
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
        chk.run_bench("nbody",          f"{B}/nbody.py",               10,       3,       None,    3,       None,    wasm_float_tol=True)
        chk.run_bench("fannkuch",       f"{B}/fannkuch.py",            30,       1,       5,       2,       None)

    if not args.no_synthetic:
        print()
        chk.run_synthetic_suite()

    rc = chk.print_summary()
    sys.exit(rc)


if __name__ == "__main__":
    main()
