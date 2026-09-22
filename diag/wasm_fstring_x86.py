#!/usr/bin/env python3
"""Throwaway diagnostic: wasm vs dynasm on str_fstring and nearby fixtures.

Stdlib only. Exit 0 always: a timed-out or failing child is reported and the
rest of the run continues. Everything printed is also written to
<out>/summary.txt.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

# check.py:child_env_base keeps these names and drops the rest, so a CI
# runner's extra variables do not move the nursery the way an inherited
# environment does.
ENV_ALLOWLIST = (
    "PATH",
    "PATHEXT",
    "COMSPEC",
    "SHELL",
    "TEMP",
    "TMP",
    "TMPDIR",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
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
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PYTHONIOENCODING",
)
# check.py:ENV_ALLOWLIST_PREFIXES
ENV_ALLOWLIST_PREFIXES = ("PYRE_", "MAJIT_", "PYPY_", "PYTHON")

TIMEOUT_S = 300

# Order named by the diagnostic: the merged bench, then the six slices.
FSTR_NAMES = (
    "fstring_simple",
    "fstring_multi",
    "fstring_spec",
    "convert_value",
    "string_ops",
    "bytes_ops",
)
CONTROL_NAMES = (
    "foriter_format_with_spec",
    "list_insert",
    "while_is_none",
)
PROFILE_LABELS = (
    "str_fstring",
    "fstr/fstring_spec",
    "fstr/convert_value",
    "list_insert",
)
PERF_EVENTS = (
    "task-clock",
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
    "page-faults",
    "context-switches",
)
_CENSUS_MS = re.compile(r"\bms=([0-9]*\.?[0-9]+)")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import gp_summary  # noqa: E402  diag/gp_summary.py, next to this script


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class Run:
    __slots__ = ("stdout", "stderr", "wall", "user", "sys_time", "rc")

    def __init__(self, stdout, stderr, wall, user, sys_time, rc):
        self.stdout = stdout
        self.stderr = stderr
        self.wall = wall
        self.user = user
        self.sys_time = sys_time
        self.rc = rc


def child_env_base():
    """Mirror check.py:child_env_base, including the PYRE_CHECK_INHERIT_ENV escape."""
    if os.environ.get("PYRE_CHECK_INHERIT_ENV"):
        env = dict(os.environ)
    else:
        env = {}
        for name, value in os.environ.items():
            upper = name.upper()
            if upper in ENV_ALLOWLIST or upper.startswith(ENV_ALLOWLIST_PREFIXES):
                env[name] = value
    # check.py:child_env_base
    env.setdefault("PYTHONIOENCODING", "utf-8:surrogateescape")
    return env


def pyre_env(repo: Path) -> dict:
    """Child environment check.py:pyre_env builds for a timed run.

    Assignments below cite the function that sets them:
    - MAJIT_STRICT, MAJIT_STATS, PYRE_DESCR_SPELLING_GATE: check.py:pyre_env
    - PYPY_GC_NURSERY (4 MiB), PYPY_GC_MIN (1 GiB): check.py:pyre_env
    - PYTHONSAFEPATH, PYTHONDONTWRITEBYTECODE: check.py:pyre_env
    - PYRE_STDLIB: check.py:_detect_pyre_stdlib via check.py:pyre_env
      (in-tree answer is <repo>/lib-python/3)
    - PYRE_WASM_MODULE, PYRE_WASM_ENGINE: check.py:pyre_env
      (WASM_MODULE_PATH resolved, WASM_ENGINE = "wasmtime")
    The two wasm keys are always this tree's module and wasmtime, which is
    the value that function writes when they are unset.
    """
    env = child_env_base()
    env["MAJIT_STRICT"] = "1"
    env["MAJIT_STATS"] = "1"
    env["PYRE_DESCR_SPELLING_GATE"] = "1"
    env.setdefault("PYPY_GC_NURSERY", str(4 * 1024 * 1024))
    env.setdefault("PYPY_GC_MIN", str(1024 * 1024 * 1024))
    env.setdefault("PYTHONSAFEPATH", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    stdlib = repo / "lib-python" / "3"
    if "PYRE_STDLIB" not in env and stdlib.is_dir():
        env["PYRE_STDLIB"] = str(stdlib)
    module = (
        repo
        / "target"
        / "wasm32-unknown-unknown"
        / "release"
        / "pyre_wasm.wasm-host.wasm"
    ).resolve()
    env["PYRE_WASM_MODULE"] = str(module)
    env["PYRE_WASM_ENGINE"] = "wasmtime"
    return env


def fixtures(repo: Path):
    rows = [("str_fstring", repo / "pyre" / "bench" / "synth" / "str_fstring.py")]
    fstr_dir = SCRIPT_DIR / "fstr"
    for name in FSTR_NAMES:
        rows.append((f"fstr/{name}", fstr_dir / f"{name}.py"))
    synth = repo / "pyre" / "bench" / "synth"
    for name in CONTROL_NAMES:
        rows.append((name, synth / f"{name}.py"))
    return rows


def _argv_text(argv):
    return " ".join(shlex.quote(str(part)) for part in argv)


def _status_rc(status):
    # Same decoding as check.py:_run_timed_unix.
    return -(status & 0x7F) if status & 0x7F else (status >> 8)


def _tail(text, n):
    lines = text.strip().splitlines()
    return lines[-n:]


def run_child(argv, env, cwd, timeout=TIMEOUT_S):
    """Run argv. Wall time is perf_counter; user/sys come from os.wait4.

    Mirrors check.py:_run_timed_unix: temp files instead of pipes, so wait4
    reaps the child, and a timeout is rc 124. Prints a failure and returns;
    the caller keeps going.
    """
    timed_out = False
    try:
        proc = None
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            try:
                proc = subprocess.Popen(
                    [str(part) for part in argv],
                    stdout=out_f,
                    stderr=err_f,
                    env=env,
                    cwd=str(cwd) if cwd is not None else None,
                    start_new_session=True,
                )
            except OSError as exc:
                print(f"FAIL spawn {_argv_text(argv)}: {exc}")
                return Run("", str(exc), 0.0, 0.0, 0.0, 127)

            def _kill():
                nonlocal timed_out
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    try:
                        proc.kill()
                    except OSError:
                        pass

            timer = threading.Timer(timeout, _kill)
            timer.daemon = True
            t0 = time.perf_counter()
            timer.start()
            try:
                _pid, status, usage = os.wait4(proc.pid, 0)
            except OSError as exc:
                _kill()
                print(f"FAIL wait4 {_argv_text(argv)}: {exc}")
                return Run("", str(exc), time.perf_counter() - t0, 0.0, 0.0, 127)
            finally:
                timer.cancel()
            wall = time.perf_counter() - t0
            rc = _status_rc(status)
            proc.returncode = rc
            out_f.seek(0)
            err_f.seek(0)
            stdout = out_f.read().decode("utf-8", "replace")
            stderr = err_f.read().decode("utf-8", "replace")
            user = max(getattr(usage, "ru_utime", 0.0), 0.0)
            sys_time = max(getattr(usage, "ru_stime", 0.0), 0.0)
    except Exception as exc:
        if proc is not None and proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        print(f"FAIL {_argv_text(argv)}: {exc}")
        return Run("", str(exc), 0.0, 0.0, 0.0, 127)

    shown = _argv_text(argv)
    if timed_out:
        print(f"TIMEOUT {shown}")
        for line in _tail(stderr, 15):
            print(line)
        return Run(stdout, stderr, wall, 0.0, 0.0, 124)
    if rc != 0:
        print(f"FAIL rc={rc} {shown}")
        for line in _tail(stderr, 20):
            print(line)
    return Run(stdout, stderr, wall, user, sys_time, rc)


def host_cmd(argv):
    exe = argv[0]
    if shutil.which(exe) is None:
        print(f"skip {exe}: not found")
        return
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"skip {exe}: {exc}")
        return
    print(f"$ {' '.join(argv)}")
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        print(f"skip {exe}: rc={proc.returncode} {err}".rstrip())


def section_system():
    print("=== 1. system ===")
    host_cmd(["uname", "-a"])
    host_cmd(["lscpu"])
    host_cmd(["nproc"])
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        print("$ /proc/meminfo")
        print("\n".join(meminfo.read_text(errors="replace").splitlines()[:3]))
    else:
        print("skip /proc/meminfo: not found")


def perf_available():
    """True when `perf stat -- true` exits 0. Anything else skips section 5."""
    try:
        proc = subprocess.run(
            ["perf", "stat", "--", "true"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _ratio(num, den):
    if num is None or den is None:
        return "n/a"
    if den == 0:
        return "inf" if num != 0 else "nan"
    return f"{num / den:.3f}"


def _agg(samples):
    if not samples:
        return None
    walls = [sample.wall for sample in samples]
    users = [sample.user for sample in samples]
    syss = [sample.sys_time for sample in samples]
    return {
        "wall_min": min(walls),
        "wall_med": statistics.median(walls),
        "user_med": statistics.median(users),
        "sys_med": statistics.median(syss),
    }


def _print_table(rows):
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cells.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        print("  ".join(cells))


def section_timing(repo, env, bins, reps, do_perfmap_warmup):
    print("=== 2. timing ===")
    wasm_bin, dynasm_bin = bins
    # Untimed: loads the main module so later runs hit the .cwasm cache.
    warm_argv = [str(wasm_bin), "--engine", "wasmtime", os.devnull]
    warm = run_child(warm_argv, env, repo)
    print(f"warmup wasm {os.devnull} rc={warm.rc} wall={warm.wall:.4f}")
    if do_perfmap_warmup:
        # main.rs:run reads PYRE_WASM_PERFMAP. Separate warmup because a
        # profiling engine may not reuse the plain .cwasm.
        perf_env = dict(env)
        perf_env["PYRE_WASM_PERFMAP"] = "1"
        warm_perf = run_child(warm_argv, perf_env, repo)
        print(
            f"warmup wasm PYRE_WASM_PERFMAP=1 {os.devnull} "
            f"rc={warm_perf.rc} wall={warm_perf.wall:.4f}"
        )

    table = [[
        "fixture",
        "dyn_wall_min",
        "dyn_wall_med",
        "wasm_wall_min",
        "wasm_wall_med",
        "wall_ratio",
        "dyn_user_med",
        "wasm_user_med",
        "user_ratio",
        "wasm_sys_med",
    ]]
    mismatches = []
    for label, path in fixtures(repo):
        if not path.is_file():
            print(f"FAIL missing fixture {path}")
            mismatches.append(label)
            table.append([label] + ["n/a"] * 9)
            continue
        collected = {}
        rcs = {}
        outs = {}
        for backend, argv in (
            ("dynasm", [str(dynasm_bin), str(path)]),
            ("wasm", [str(wasm_bin), "--engine", "wasmtime", str(path)]),
        ):
            samples = []
            backend_rcs = []
            backend_outs = []
            for i in range(1, reps + 1):
                result = run_child(argv, env, repo)
                backend_rcs.append(result.rc)
                print(
                    f"run {label} {backend} {i}/{reps} rc={result.rc} "
                    f"wall={result.wall:.4f} user={result.user:.4f} "
                    f"sys={result.sys_time:.4f}"
                )
                if result.rc == 0:
                    samples.append(result)
                    backend_outs.append(result.stdout)
            collected[backend] = samples
            rcs[backend] = backend_rcs
            outs[backend] = backend_outs
        all_rcs = rcs["dynasm"] + rcs["wasm"]
        all_outs = outs["dynasm"] + outs["wasm"]
        rc_ok = bool(all_rcs) and all(rc == 0 for rc in all_rcs)
        out_ok = bool(all_outs) and all(text == all_outs[0] for text in all_outs)
        if not (rc_ok and out_ok):
            mismatches.append(label)
        dyn = _agg(collected["dynasm"])
        wasm = _agg(collected["wasm"])
        table.append([
            label,
            _fmt(None if dyn is None else dyn["wall_min"]),
            _fmt(None if dyn is None else dyn["wall_med"]),
            _fmt(None if wasm is None else wasm["wall_min"]),
            _fmt(None if wasm is None else wasm["wall_med"]),
            _ratio(
                None if wasm is None else wasm["wall_min"],
                None if dyn is None else dyn["wall_min"],
            ),
            _fmt(None if dyn is None else dyn["user_med"]),
            _fmt(None if wasm is None else wasm["user_med"]),
            _ratio(
                None if wasm is None else wasm["user_med"],
                None if dyn is None else dyn["user_med"],
            ),
            _fmt(None if wasm is None else wasm["sys_med"]),
        ])
    print()
    _print_table(table)
    for label in mismatches:
        print(f"MISMATCH {label}")


def _prefixed(stderr, prefix):
    lines = []
    for line in stderr.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(prefix):
            lines.append(line)
    return lines


def _census_to_print(stderr):
    """Summary [compile-census] lines, then the 10 slowest module lines.

    main.rs:jit_compile_trace prints one `[compile-census] kind=... ms=...`
    line per module and no aggregate. Lines without kind=/ms= are treated as
    summaries if a build ever emits them.
    """
    summaries = []
    modules = []
    for idx, line in enumerate(stderr.splitlines()):
        if "[compile-census]" not in line.lstrip():
            continue
        if "kind=" in line and _CENSUS_MS.search(line):
            ms = float(_CENSUS_MS.search(line).group(1))
            modules.append((ms, idx, line))
        else:
            summaries.append(line)
    modules.sort(key=lambda item: (-item[0], item[1]))
    slowest = [line for _ms, _idx, line in modules[:10]]
    return summaries, slowest, len(modules)


def section_jit(repo, env, bins):
    """Wasm JIT readout and dynasm [jit-stats] lines.

    PYRE_WASM_JIT_STATS (main.rs:run) prints `[jit-stats]` lines, including
    `compiles=... compile_ms=...`. `nofuel` keeps that readout without fuel.
    PYRE_WASM_COMPILE_CENSUS (main.rs:jit_compile_trace) prints
    `[compile-census]` lines. MAJIT_STATS (pyrex:maybe_print_jit_stats, also
    set by check.py:pyre_env) prints `[jit-stats]` lines on dynasm.
    """
    print("=== 3. jit stats ===")
    wasm_bin, dynasm_bin = bins
    chosen = [
        (label, path)
        for label, path in fixtures(repo)
        if label == "str_fstring" or label.startswith("fstr/")
    ]
    for label, path in chosen:
        if not path.is_file():
            print(f"FAIL missing fixture {path}")
            continue
        nofuel_env = dict(env)
        # main.rs:run — PYRE_WASM_JIT_STATS=nofuel
        nofuel_env["PYRE_WASM_JIT_STATS"] = "nofuel"
        nofuel = run_child(
            [str(wasm_bin), "--engine", "wasmtime", str(path)],
            nofuel_env,
            repo,
        )
        print(f"-- {label} wasm PYRE_WASM_JIT_STATS=nofuel rc={nofuel.rc} --")
        stats_lines = _prefixed(nofuel.stderr, "[jit-stats]")
        if stats_lines:
            for line in stats_lines:
                print(line)
        else:
            print("(no [jit-stats] lines)")

        census_env = dict(env)
        # main.rs:jit_compile_trace — PYRE_WASM_COMPILE_CENSUS
        census_env["PYRE_WASM_COMPILE_CENSUS"] = "1"
        census = run_child(
            [str(wasm_bin), "--engine", "wasmtime", str(path)],
            census_env,
            repo,
        )
        summaries, slowest, n_modules = _census_to_print(census.stderr)
        print(
            f"-- {label} wasm PYRE_WASM_COMPILE_CENSUS=1 rc={census.rc} "
            f"({len(slowest)} slowest of {n_modules}) --"
        )
        if summaries:
            for line in summaries:
                print(line)
        if slowest:
            for line in slowest:
                print(line)
        elif not summaries:
            print("(no [compile-census] lines)")

        # pyrex:maybe_print_jit_stats — prefix `[jit-stats]`
        dyn_env = dict(env)
        dyn_env["MAJIT_STATS"] = "1"
        dyn = run_child([str(dynasm_bin), str(path)], dyn_env, repo)
        print(f"-- {label} dynasm MAJIT_STATS=1 rc={dyn.rc} --")
        dyn_lines = _prefixed(dyn.stderr, "[jit-stats]")
        if dyn_lines:
            for line in dyn_lines:
                print(line)
        else:
            print("(no [jit-stats] lines)")


def _profile_filename(label):
    return "gp-" + label.replace("/", "-") + ".json"


def section_guest(repo, env, bins, out_dir):
    """main.rs:run writes PYRE_WASM_GUEST_PROFILE as Firefox processed JSON."""
    print("=== 4. guest profile ===")
    wasm_bin, _dynasm_bin = bins
    by_label = dict(fixtures(repo))
    for label in PROFILE_LABELS:
        path = by_label.get(label)
        if path is None or not path.is_file():
            print(f"FAIL missing fixture {label}")
            continue
        dest = (out_dir / _profile_filename(label)).resolve()
        profile_env = dict(env)
        profile_env["PYRE_WASM_GUEST_PROFILE"] = str(dest)
        result = run_child(
            [str(wasm_bin), "--engine", "wasmtime", str(path)],
            profile_env,
            repo,
        )
        print(f"-- {label} {dest} rc={result.rc} --")
        if result.rc != 0 or not dest.is_file():
            print(f"FAIL guest profile {label}: no profile written")
            continue
        try:
            gp_summary.main(str(dest), 30)
        except Exception as exc:
            print(f"FAIL gp_summary {dest}: {exc}")
            traceback.print_exc()


def _parse_perf_stat(text):
    counters = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(",")
        if len(parts) < 3:
            continue
        raw = parts[0].strip()
        event = parts[2].strip()
        if not event:
            continue
        if raw.startswith("<"):
            counters[event] = None
            continue
        try:
            counters[event] = float(raw)
        except ValueError:
            continue
    return counters


def _counter(counters, name):
    if name in counters:
        return counters[name]
    for key, value in counters.items():
        if key == name or key.startswith(name + ":"):
            return value
    return None


def _ipc(counters):
    cycles = _counter(counters, "cycles")
    instr = _counter(counters, "instructions")
    if cycles in (None, 0) or instr is None:
        return None
    return instr / cycles


def _print_perf_report(data_path, extra, out_path, limit):
    cmd = ["perf", "report", "-i", str(data_path), *extra]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        text = f"FAIL {' '.join(cmd)}: {exc}\n"
        out_path.write_text(text, encoding="utf-8")
        print(f"-- {out_path.name} --")
        print(text, end="" if text.endswith("\n") else "\n")
        return
    text = proc.stdout or ""
    if proc.returncode != 0:
        text += proc.stderr or ""
    out_path.write_text(text, encoding="utf-8")
    print(f"-- {out_path.name} (first {limit}) --")
    body = text.splitlines()
    if not body:
        print("(empty)")
        return
    for line in body[:limit]:
        print(line)


def section_perf(repo, env, bins, out_dir):
    print("=== 5. perf ===")
    wasm_bin, dynasm_bin = bins
    by_label = dict(fixtures(repo))
    targets = ("str_fstring", "list_insert")
    parsed = {}
    for label in targets:
        path = by_label[label]
        parsed[label] = {}
        for backend, argv, extra_env in (
            ("dynasm", [str(dynasm_bin), str(path)], {}),
            ("wasm", [str(wasm_bin), "--engine", "wasmtime", str(path)], {}),
        ):
            run_env = dict(env)
            run_env.update(extra_env)
            cmd = [
                "perf",
                "stat",
                "-x,",
                "-e",
                ",".join(PERF_EVENTS),
                "--",
                *argv,
            ]
            result = run_child(cmd, run_env, repo)
            print(f"-- perf stat {label} {backend} rc={result.rc} --")
            blob = result.stderr if result.stderr.strip() else result.stdout
            for line in blob.splitlines():
                if line.strip():
                    print(line)
            counters = _parse_perf_stat(blob)
            parsed[label][backend] = counters
            ipc = _ipc(counters)
            print(f"IPC {label} {backend} {_ratio(ipc, 1.0) if ipc is not None else 'n/a'}")
        dyn_c = parsed[label].get("dynasm", {})
        wasm_c = parsed[label].get("wasm", {})
        print(
            f"ratio {label} wasm/dynasm "
            f"cycles={_ratio(_counter(wasm_c, 'cycles'), _counter(dyn_c, 'cycles'))} "
            f"instructions={_ratio(_counter(wasm_c, 'instructions'), _counter(dyn_c, 'instructions'))}"
        )

    script = by_label["str_fstring"]
    for backend, argv, extra in (
        ("dynasm", [str(dynasm_bin), str(script)], {}),
        (
            "wasm",
            [str(wasm_bin), "--engine", "wasmtime", str(script)],
            # main.rs:run — names JIT functions in the perf map.
            {"PYRE_WASM_PERFMAP": "1"},
        ),
    ):
        data = out_dir / f"perf-{backend}-str_fstring.data"
        run_env = dict(env)
        run_env.update(extra)
        cmd = [
            "perf",
            "record",
            "-F",
            "1999",
            "-g",
            "-o",
            str(data),
            "--",
            *argv,
        ]
        result = run_child(cmd, run_env, repo)
        print(f"-- perf record {backend} str_fstring rc={result.rc} --")
        if result.rc != 0 or not data.is_file():
            print(f"FAIL perf record {backend}: no data file")
            continue
        reports = (
            (
                "dso",
                ["--stdio", "--no-children", "--sort", "dso"],
                60,
            ),
            (
                "dso-sym",
                ["--stdio", "--no-children", "--sort", "dso,sym"],
                100,
            ),
            (
                "children-sym",
                ["--stdio", "--children", "--sort", "sym"],
                60,
            ),
        )
        for kind, extra_args, limit in reports:
            out_path = out_dir / f"perf-{backend}-str_fstring-{kind}.txt"
            _print_perf_report(data, extra_args, out_path, limit)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", type=Path, help="repo root (default cwd)")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--reps", type=int, default=5, help="timed repetitions (default 5)")
    return parser.parse_args(argv)


def run_all(args):
    repo = args.repo.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.reps < 1:
        print(f"FAIL --reps must be >= 1 (got {args.reps})")
        return
    env = pyre_env(repo)
    bins = (
        repo / "target" / "release" / "pyre-wasm-runner",
        repo / "target" / "release" / "pyre-dynasm",
    )
    for binary in bins:
        if not binary.is_file():
            print(f"FAIL missing binary {binary}")
    section_system()
    do_perf = perf_available()
    try:
        section_timing(repo, env, bins, args.reps, do_perf)
    except Exception:
        traceback.print_exc()
    try:
        section_jit(repo, env, bins)
    except Exception:
        traceback.print_exc()
    try:
        section_guest(repo, env, bins, out_dir)
    except Exception:
        traceback.print_exc()
    try:
        if do_perf:
            section_perf(repo, env, bins, out_dir)
        else:
            print("=== 5. perf ===")
            print("perf unavailable")
    except Exception:
        traceback.print_exc()


def main(argv=None):
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = open(out_dir / "summary.txt", "w", encoding="utf-8")
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(stdout, summary)
    sys.stderr = Tee(stderr, summary)
    try:
        run_all(args)
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        summary.close()
        sys.stdout = stdout
        sys.stderr = stderr
    return 0


if __name__ == "__main__":
    sys.exit(main())
