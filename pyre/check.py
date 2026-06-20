#!/usr/bin/env python3
"""pyre pre-merge check: correctness + regression guard + comparison

Cross-platform Python translation of pyre/check.sh.
"""

import argparse
import os
import shutil
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
}

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

# ── Check runner ─────────────────────────────────────────────────────

class Check:
    def __init__(self, args):
        self.args = args
        self.results = []
        self.comparisons = []
        self.dynasm_pass = self.dynasm_fail = 0
        self.cranelift_pass = self.cranelift_fail = 0
        self.dynasm_pyre = ""
        self.cranelift_pyre = ""
        self.snapshot_diffs = []
        self.snapshot_missing = []

    # ── backend helpers ──

    def enabled(self, backend):
        return bool(self._pyre(backend))

    def _pyre(self, backend):
        return self.dynasm_pyre if backend == "dynasm" else self.cranelift_pyre

    def _timeout_scale(self, backend):
        if backend == "dynasm" and self.args.dynasm_timeout_scale is not None:
            return self.args.dynasm_timeout_scale
        if backend == "cranelift" and self.args.cranelift_timeout_scale is not None:
            return self.args.cranelift_timeout_scale
        return self.args.timeout_scale

    def _set_pyre(self, backend, path):
        if backend == "dynasm":
            self.dynasm_pyre = path
        else:
            self.cranelift_pyre = path

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
                "dynasm": "-",
                "cranelift": "-",
            }
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
            if backend == "dynasm":
                self.dynasm_pass += 1
            else:
                self.cranelift_pass += 1
        else:
            self.results.append(f"{red('FAIL')} {backend} {name}  {detail}")
            if backend == "dynasm":
                self.dynasm_fail += 1
            else:
                self.cranelift_fail += 1

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
            if "no LLBC source resolved" in (proc.stderr or ""):
                # The JIT front-end has no MIR to lower because build/llbc/
                # is empty. This is a setup step, not a toolchain fault — the
                # rustup diagnostics below would be noise, so point at the
                # producer instead.
                print(red("LLBC artefacts are missing under build/llbc/."))
                print("Run the extractor first, then re-run this script:")
                print("    scripts/extract-llbc.py")
            else:
                self._print_cargo_diagnostics(cargo_path)
            sys.exit(1)
        lines = (proc.stdout or "").strip().splitlines() + (proc.stderr or "").strip().splitlines()
        if lines:
            print(lines[-1])

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
        for backend in ("dynasm", "cranelift"):
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

        if output != pypy_output:
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
        skip_backends=(),
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
            for backend in ("dynasm", "cranelift"):
                if self.enabled(backend):
                    self._record(backend, False, name, "pypy crash")
                    self._append_comparison(backend, name, t_cpython, "-", "FAIL")
            return
        print(f"{dim('done')}  {t_pypy}s")

        for backend, vs_cpython, vs_pypy in [
            ("dynasm", dynasm_vs_cpython, dynasm_vs_pypy),
            ("cranelift", cranelift_vs_cpython, cranelift_vs_pypy),
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
            for backend in ("dynasm", "cranelift"):
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
            for backend in ("dynasm", "cranelift"):
                if self.enabled(backend):
                    self._record(backend, False, name, "pypy crash")
                    self._append_comparison(backend, name, t_cpython, "-", "FAIL")
            return
        t_pypy = f"{pypy_time:.2f}"
        print(f"{dim('done')}  {t_pypy}s")

        if cpython_output is not None and cpython_output != pypy_output:
            print(f"    {'baseline':<10s}{red('WRONG')}  cpython output differs from pypy")
            for backend in ("dynasm", "cranelift"):
                if self.enabled(backend):
                    self._record(backend, False, name, "cpython/pypy output mismatch")
                    self._append_comparison(
                        backend, name, t_cpython, t_pypy, "BASEFAIL",
                    )
            return

        for backend in ("dynasm", "cranelift"):
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
        for b in ("dynasm", "cranelift"):
            if self.enabled(b):
                parts.append(f"{b}={self._pyre(b)}(x{self._timeout_scale(b)})")
        if parts:
            print(f"backend: {' '.join(parts)}")

    def print_comparison_table(self):
        if not self.comparisons:
            return
        both = self.enabled("dynasm") and self.enabled("cranelift")
        dynasm_only = self.enabled("dynasm") and not self.enabled("cranelift")
        cranelift_only = self.enabled("cranelift") and not self.enabled("dynasm")

        print(bold("Comparison"))

        if both:
            print(f"  {'benchmark':<35s} {'cpython':>8s} {'pypy':>8s} {'dynasm':>18s} {'cranelift':>18s}")
            print("  " + "─" * 98)
            for c in self.comparisons:
                print(
                    f"  {c['name']:<35s} {c['cpython']:>8s} {c['pypy']:>8s}"
                    f" {c['dynasm']:>18s} {c['cranelift']:>18s}"
                )
        elif dynasm_only:
            print(f"  {'benchmark':<35s} {'cpython':>8s} {'pypy':>8s} {'dynasm':>18s}")
            print("  " + "─" * 76)
            for c in self.comparisons:
                print(
                    f"  {c['name']:<35s} {c['cpython']:>8s} {c['pypy']:>8s}"
                    f" {c['dynasm']:>18s}"
                )
        elif cranelift_only:
            print(f"  {'benchmark':<35s} {'cpython':>8s} {'pypy':>8s} {'cranelift':>18s}")
            print("  " + "─" * 76)
            for c in self.comparisons:
                print(
                    f"  {c['name']:<35s} {c['cpython']:>8s} {c['pypy']:>8s}"
                    f" {c['cranelift']:>18s}"
                )

    def print_summary(self):
        print()
        if self.results:
            print("─" * 53)
            for r in self.results:
                print(f"  {r}")
            print("─" * 53)

        failed_runs = 0
        enabled_runs = 0
        for b in ("dynasm", "cranelift"):
            if not self.enabled(b):
                continue
            enabled_runs += 1
            fail = self.dynasm_fail if b == "dynasm" else self.cranelift_fail
            if fail > 0:
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

        for b in ("dynasm", "cranelift"):
            if not self.enabled(b):
                continue
            p = self.dynasm_pass if b == "dynasm" else self.cranelift_pass
            f = self.dynasm_fail if b == "dynasm" else self.cranelift_fail
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

    parser = argparse.ArgumentParser(
        description="pyre pre-merge check: correctness + regression guard + comparison",
        allow_abbrev=False,
    )
    parser.add_argument("--backend", choices=["dynasm", "cranelift"], default="")
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument("--dynasm-timeout-scale", type=float, default=None)
    parser.add_argument("--cranelift-timeout-scale", type=float, default=None)
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

    if args.pyre_path and not args.backend:
        parser.error("[path/to/pyre] requires --backend when running a single binary")

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
    chk = Check(args)

    backends = [args.backend] if args.backend else ["dynasm", "cranelift"]

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

        #             name              script                          timeout  d_vs_cp  d_vs_py  c_vs_cp  c_vs_py  skip
        chk.run_bench("int_loop",       f"{B}/int_loop.py",             5,       None,    2,       None,    2)
        chk.run_bench("float_loop",     f"{B}/float_loop.py",           5,       None,    1.5,     None,    1.5)
        chk.run_bench("fib_loop",       f"{B}/fib_loop.py",             5,       2,       4,       2,       4)
        chk.run_bench("inline_helper",  f"{B}/inline_helper.py",        5,       None,    1.2,     None,    1.2)
        chk.run_bench("fib_recursive",  f"{B}/fib_recursive.py",        5,       2,       13,      2,       13)
        chk.run_bench("nested_loop",    f"{B}/nested_loop.py",          5,       None,    2,       None,    3)
        chk.run_bench("raise_catch",    f"{B}/raise_catch_loop.py",     5,       None,    1.5,       None,    2)
        chk.run_bench("spectral_norm",  f"{B}/spectral_norm.py",        5,       2,       7,       2,       7)
        chk.run_bench("nbody",          f"{B}/nbody.py",               10,       3,       None,    3,       None)
        chk.run_bench("fannkuch",       f"{B}/fannkuch.py",            30,       1,       5,       2,       None)
        chk.run_bench("list_reverse",   f"{B}/list_reverse.py",         5,       15,      None,    15,      None)
        chk.run_bench("list_pop_append",f"{B}/list_pop_append.py",      5,       30,      None,    30,      None)
        chk.run_bench("list_insert",    f"{B}/list_insert.py",          5,       None,    2,       None,    2)
        chk.run_bench("list_setslice",  f"{B}/list_setslice.py",        5,       15,      None,    15,      None)

    if not args.no_synthetic:
        print()
        chk.run_synthetic_suite()

    rc = chk.print_summary()
    sys.exit(rc)


if __name__ == "__main__":
    main()
