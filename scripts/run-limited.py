#!/usr/bin/env python3
"""Run one build/test tree with bounded RSS, output, time and concurrency.

The RSS watchdog is a sampled safety net, not a kernel aggregate-memory quota.
For hostile or instantaneous allocations use a memory-capped VM/container too.
No command is executed through a shell. Logs are retained outside process RAM.
"""

import argparse
import ctypes
import fcntl
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
import tempfile
import time

MIB = 1024 * 1024


def process_info(pid):
    # proc_pid_stat(5): parse after the last ')' because comm can contain spaces.
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[1]), int(fields[2]),
                int(fields[21]) * os.sysconf("SC_PAGE_SIZE"), int(fields[19]))
    except (OSError, ValueError, IndexError):
        return None


def process_tree(root, seen=None):
    rows = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit() and (info := process_info(int(path.name))) is not None:
            rows[int(path.name)] = info
    selected = {pid for pid, (parent, group, _, _) in rows.items()
                if group == root or parent == os.getpid()}
    selected.add(root)
    # Historical descendants can detach, but a recycled PID is another process.
    for pid, old in (seen or {}).items():
        if pid in rows and rows[pid][3] == old[3]:
            selected.add(pid)
    while True:
        more = {pid for pid, (parent, _, _, _) in rows.items()
                if pid not in selected and parent in selected}
        if not more:
            break
        selected.update(more)
    return {pid: rows[pid] for pid in selected if pid in rows}


def become_subreaper():
    # PR_SET_CHILD_SUBREAPER: even a child detached before our first sample
    # is adopted here when its parent exits (prctl(2)).
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, ctypes.c_ulong(1), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def adopted_child_pids():
    # `/proc/self/task/<pid>/children` needs CONFIG_CHECKPOINT_RESTORE.
    # Without that file, scan `/proc` for processes whose ppid is us.
    children_path = Path(f"/proc/self/task/{os.getpid()}/children")
    try:
        return [int(pid) for pid in children_path.read_text().split()]
    except FileNotFoundError:
        me = os.getpid()
        try:
            entries = list(Path("/proc").iterdir())
        except OSError:
            return []
        found = []
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat = Path(f"/proc/{entry.name}/stat").read_text()
            except OSError:
                continue
            # `pid (comm) state ppid ...`; comm may contain spaces and ')'.
            try:
                after = stat.rsplit(")", 1)[1].split()
                ppid = int(after[1])
            except (IndexError, ValueError):
                continue
            if ppid == me:
                found.append(int(entry.name))
        return found


def stop_tree(root):
    # The root has not been reaped yet: its process-group ID cannot be reused.
    for sig in (signal.SIGSTOP, signal.SIGKILL):
        try:
            os.killpg(root, sig)
        except OSError:
            pass


def cleanup(proc):
    # Group cleanup must run without depending on a successful process census.
    try:
        stop_tree(proc.pid)
    finally:
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait()
    # Detached descendants are now our children. Signal only these unreaped
    # children, never historical PID numbers. Their PIDs cannot be recycled
    # until waitpid below; repeat to collect descendants adopted as they die.
    while children := adopted_child_pids():
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        for pid in children:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-mib", type=int, default=2048)
    parser.add_argument("--process-mib", type=int, default=1024)
    parser.add_argument("--address-space-mib", type=int,
                        help="virtual reservation ceiling; defaults to process-mib (VM RAM cap still required)")
    parser.add_argument("--output-mib", type=int, default=16)
    parser.add_argument("--file-mib", type=int,
                        help="per-file ceiling for build artifacts; defaults to output-mib")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("run inside a RAM-capped Linux VM/container")
    # getrlimit(2): RLIMIT_DATA covers mmap allocations starting with Linux 4.7.
    version = tuple(map(int, re.match(r"(\d+)\.(\d+)", os.uname().release).groups()))
    if version < (4, 7):
        parser.error("Linux 4.7 or newer is required for the allocation budget")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or min(args.memory_mib, args.process_mib, args.output_mib, args.seconds) <= 0:
        parser.error("supply a command and positive limits")
    if args.memory_mib > 8192 or args.process_mib > args.memory_mib:
        parser.error("tree budget must be <= 8192 MiB; process budget must fit in it")
    address_space_mib = args.address_space_mib if args.address_space_mib is not None else args.process_mib
    if not args.process_mib <= address_space_mib <= 65536:
        parser.error("address-space budget must be between process-mib and 65536 MiB")
    file_mib = args.file_mib if args.file_mib is not None else args.output_mib
    if not 0 < file_mib <= 4096:
        parser.error("file budget must be between 1 and 4096 MiB")
    lock_path = Path(tempfile.gettempdir()) / f"majit-safe-run-{os.getuid()}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another protected build/test is running; run sequentially", file=sys.stderr)
            return 125
        directory = Path(tempfile.mkdtemp(prefix="majit-limited-"))
        print(f"logs: {directory}; tree={args.memory_mib} MiB, process={args.process_mib} MiB", flush=True)

        def limits():
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_DATA, (args.process_mib * MIB,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_mib * MIB,) * 2)
            # Address space bounds even an allocation between RSS samples.
            if sys.platform.startswith("linux"):
                resource.setrlimit(resource.RLIMIT_AS, (address_space_mib * MIB,) * 2)

        env = dict(os.environ, CARGO_BUILD_JOBS="1", RUST_TEST_THREADS="1", LLBC_PARALLEL_LAYOUTS="0")
        # The shared sccache server can spawn compilers outside this process
        # tree; direct rustc keeps the memory budget and descendants observable.
        env.pop("RUSTC_WRAPPER", None)
        env.pop("RUSTC_WORKSPACE_WRAPPER", None)
        env["CARGO_BUILD_RUSTC_WRAPPER"] = ""
        reason = None
        peak = 0
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr, (directory / "rss.tsv").open("w", buffering=1) as samples:
            samples.write("seconds\tpid\tppid\trss_bytes\n")
            become_subreaper()
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    env=env, start_new_session=True, preexec_fn=limits)
            started = time.monotonic()
            members = {}
            seen = {}
            next_sample = 0.0
            sample_bytes = 0
            try:
                # WNOWAIT keeps the root PID reserved until cleanup completes.
                while os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                    members = process_tree(proc.pid, seen)
                    seen.update(members)
                    rss = sum(row[2] for row in members.values())
                    peak = max(peak, rss)
                    elapsed = time.monotonic() - started
                    # Preserve the allocating PID and growth curve, not just
                    # a final aggregate peak. Bound this diagnostic file too.
                    if elapsed >= next_sample and sample_bytes < MIB:
                        for pid, (parent, _, resident, _) in sorted(members.items()):
                            row = f"{elapsed:.3f}\t{pid}\t{parent}\t{resident}\n"
                            if sample_bytes + len(row) > MIB:
                                break
                            samples.write(row)
                            sample_bytes += len(row)
                        next_sample = elapsed + 0.5
                    if rss > args.memory_mib * MIB:
                        reason = "process-tree RSS limit exceeded"
                    elif any(row[2] > args.process_mib * MIB for row in members.values()):
                        reason = "process RSS limit exceeded"
                    elif time.monotonic() - started > args.seconds:
                        reason = "deadline exceeded"
                    elif os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size >= args.output_mib * MIB:
                        reason = "output limit exceeded"
                    if reason:
                        break
                    time.sleep(0.1)
            finally:
                cleanup(proc)
            code = proc.returncode
        # Bound our own memory too: never read the complete captured log.
        for path in (directory / "stdout.log", directory / "stderr.log"):
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16384))
                sys.stdout.buffer.write(stream.read(16384))
        print(f"\nexit={code}; peak tree RSS={peak / MIB:.1f} MiB; {reason or 'command finished'}", flush=True)
        return 124 if reason else (code if code >= 0 else 128 - code)


if __name__ == "__main__":
    raise SystemExit(main())
