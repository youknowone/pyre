#!/usr/bin/env python3
"""Run one build/test tree with bounded RSS, output, time and concurrency.

The RSS watchdog is a sampled safety net, not a kernel aggregate-memory quota.
For hostile or instantaneous allocations use a memory-capped VM/container too.
No command is executed through a shell. Logs are retained outside process RAM.
"""

import argparse
import fcntl
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time

MIB = 1024 * 1024


def process_tree(root, extra_pids=()):
    rows = {}
    text = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid=,rss="], text=True)
    for line in text.splitlines():
        pid, parent, group, rss = map(int, line.split())
        rows[pid] = (parent, group, rss * 1024)
    selected = {pid for pid, (_, group, _) in rows.items() if group == root}
    selected.add(root)
    selected.update(extra_pids)
    while True:
        more = {
            pid
            for pid, (parent, group, _) in rows.items()
            if pid not in selected and (parent in selected or group in selected)
        }
        if not more:
            break
        selected.update(more)
    return {pid: rows[pid] for pid in selected if pid in rows}


def stop_tree(root, members):
    # Stop before killing so a runaway cannot keep spawning while we clean up.
    for sig in (signal.SIGSTOP, signal.SIGKILL):
        try:
            os.killpg(root, sig)
        except ProcessLookupError:
            pass
        for pid in members:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
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
    if sys.platform == "darwin":
        parser.error("macOS cannot enforce this kernel memory budget; run inside a RAM-capped Linux VM/container")
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
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    env=env, start_new_session=True, preexec_fn=limits)
            started = time.monotonic()
            members = {}
            seen_pids = {proc.pid}
            next_sample = 0.0
            sample_bytes = 0
            try:
                while proc.poll() is None:
                    members = process_tree(proc.pid, seen_pids)
                    seen_pids.update(members)
                    rss = sum(row[2] for row in members.values())
                    peak = max(peak, rss)
                    elapsed = time.monotonic() - started
                    # Preserve the allocating PID and growth curve, not just
                    # a final aggregate peak. Bound this diagnostic file too.
                    if elapsed >= next_sample and sample_bytes < MIB:
                        for pid, (parent, _, resident) in sorted(members.items()):
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
                        stop_tree(proc.pid, members)
                        break
                    time.sleep(0.1)
                code = proc.wait()
            finally:
                stop_tree(proc.pid, process_tree(proc.pid, seen_pids))
                proc.wait()
        # Bound our own memory too: never read the complete captured log.
        for path in (directory / "stdout.log", directory / "stderr.log"):
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16384))
                sys.stdout.buffer.write(stream.read(16384))
        print(f"\nexit={code}; peak tree RSS={peak / MIB:.1f} MiB; {reason or 'command finished'}", flush=True)
        return 124 if reason else (code if code >= 0 else 128 - code)


if __name__ == "__main__":
    raise SystemExit(main())
