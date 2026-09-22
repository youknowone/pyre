"""DIAG (not for merge): pyre dynasm vs PyPy 7.3.23 vs PyPy 8.0.0 on the top-level benches, same runner.

Timing goes through check.py's own `run_timed` (user CPU; Windows job object) and
`pyre_env`, so the numbers are what check.py's gate sees. `--perf` (Linux) adds
`perf stat` counters and a sample-annotated disassembly of each JIT's hottest
code region, rebuilt from `MAJIT_DUMP=1` (majit-backend-dynasm runner.rs /
x86/assembler.rs CODE DUMP) and `PYPYLOG=jit-backend-dump`
(rpython/jit/backend/llsupport/asmmemmgr.py `_dump`).
"""

import argparse
import collections
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, "pyre")
import check  # noqa: E402

TIMEOUT = 600
OUT = None


def log(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    with open(os.path.join(OUT, "summary.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(argv, env=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT):
    try:
        p = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, timeout=TIMEOUT)
        out = p.stdout.decode("utf-8", "replace") if isinstance(p.stdout, bytes) else ""
        return p.returncode, out
    except (OSError, subprocess.SubprocessError) as e:
        return -1, f"{type(e).__name__}: {e}"


def section(title):
    log(f"\n=== {title} ===")


def system_info(runners):
    section("1. system")
    log("platform", platform.platform(), platform.processor())
    if sys.platform.startswith("linux"):
        for argv in (["uname", "-a"], ["lscpu"]):
            log(sh(argv)[1].rstrip())
    elif sys.platform == "win32":
        ps = "$p=Get-CimInstance Win32_Processor; $p.Name; $p.MaxClockSpeed; $p.NumberOfLogicalProcessors"
        log(sh(["powershell", "-NoProfile", "-Command", ps])[1].rstrip())
    elif sys.platform == "darwin":
        log(sh(["sysctl", "-n", "machdep.cpu.brand_string"])[1].rstrip())
    for name, argv, _ in runners:
        log(f"{name}: {' '.join(argv)} :: {sh(argv + ['--version'])[1].strip()[:200]}")


def timing(runners, benches, reps):
    section("2. timing (user CPU s, runners interleaved per rep)")
    names = [n for n, _, _ in runners]
    for bench in benches:
        path = f"pyre/bench/{bench}.py"
        times = {n: [] for n in names}
        outs = {}
        for rep in range(reps):
            for name, argv, env in runners:
                out, t, rc, _ = check.run_timed(argv + [path], timeout_s=TIMEOUT, env=env)
                if rc != 0:
                    log(f"  {bench} {name} rep{rep} rc={rc}")
                    continue
                times[name].append(t)
                outs.setdefault(name, out)
        ref = outs.get("pyre")
        for name in names[1:]:
            if ref is not None and name in outs and outs[name] != ref:
                log(f"  MISMATCH {bench} pyre vs {name}")
        cells = []
        for name in names:
            ts = times[name]
            if ts:
                cells.append(f"{name} med={statistics.median(ts):.3f} [{min(ts):.3f}..{max(ts):.3f}]")
            else:
                cells.append(f"{name} -")
        ratios = []
        if times["pyre"]:
            for name in names[1:]:
                if times[name]:
                    ratios.append(f"pyre/{name}={statistics.median(times['pyre']) / statistics.median(times[name]):.2f}x")
        log(f"{bench:<14s} " + "  ".join(cells) + "  " + " ".join(ratios))


def dump_env(name):
    return {"MAJIT_DUMP": "1"} if name == "pyre" else None


def code_dumps(runners, out_dir):
    section("3. code dumps / trace logs (float_loop)")
    path = "pyre/bench/float_loop.py"
    for name, argv, env in runners:
        if name == "pyre":
            dump = os.path.join(out_dir, "pyre-float_loop-dump.txt")
            with open(dump, "wb") as f:
                rc, _ = sh(argv + [path], env={**env, "MAJIT_DUMP": "1"}, stdout=subprocess.DEVNULL, stderr=f)
            trace = os.path.join(out_dir, "pyre-float_loop-trace.txt")
            # majit-metainterp lib.rs reads MAJIT_LOG_OPT
            with open(trace, "wb") as f:
                sh(argv + [path], env={**env, "MAJIT_LOG_OPT": "1"}, stdout=subprocess.DEVNULL, stderr=f)
            log(f"{name}: dump rc={rc} {os.path.getsize(dump)}B, trace {os.path.getsize(trace)}B")
        else:
            plog = os.path.join(out_dir, f"{name}-float_loop.pypylog")
            rc, _ = sh(argv + [path], env={**os.environ, "PYPYLOG": f"jit-backend-dump,jit-log-opt:{plog}"},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            size = os.path.getsize(plog) if os.path.exists(plog) else 0
            log(f"{name}: pypylog rc={rc} {size}B")


PYRE_HDR = re.compile(r"\[dynasm\] (?:BRIDGE )?CODE DUMP \((\d+) bytes at (0x[0-9a-fA-F]+)")
PYPY_LINE = re.compile(r"CODE_DUMP @([0-9a-fA-F]+) \+(\d+)\s+([0-9A-Fa-f]+)")


def parse_pyre_dump(text):
    """Return {addr: byte} from majit-backend-dynasm CODE DUMP blocks (LE u32 words, %08x)."""
    mem = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = PYRE_HDR.search(lines[i])
        if not m:
            i += 1
            continue
        size, base = int(m.group(1)), int(m.group(2), 16)
        words = []
        i += 1
        while i < len(lines) and len(words) * 4 < size:
            toks = lines[i].split()
            if not toks or not all(re.fullmatch(r"[0-9a-f]{8}", t) for t in toks):
                break
            words.extend(int(t, 16) for t in toks)
            i += 1
        data = b"".join(w.to_bytes(4, "little") for w in words)[:size]
        for off, b in enumerate(data):
            mem[base + off] = b
    return mem


def parse_pypy_dump(text):
    mem = {}
    for m in PYPY_LINE.finditer(text):
        base = int(m.group(1), 16) + int(m.group(2))
        data = bytes.fromhex(m.group(3))
        for off, b in enumerate(data):
            mem[base + off] = b
    return mem


def regions(mem):
    """Contiguous [start, end) runs of known bytes."""
    out = []
    for addr in sorted(mem):
        if out and out[-1][1] == addr:
            out[-1][1] = addr + 1
        else:
            out.append([addr, addr + 1])
    return [tuple(r) for r in out]


def annotate(name, mem, ips, out_dir):
    total = sum(ips.values())
    regs = regions(mem)
    log(f"-- {name}: {total} samples, {len(regs)} dump regions, {len(mem)} bytes")
    if not regs or not total:
        return
    in_region = collections.Counter()
    for (s, e) in regs:
        in_region[(s, e)] = sum(c for ip, c in ips.items() if s <= ip < e)
    (s, e), hot = in_region.most_common(1)[0]
    log(f"   hottest region {s:#x}..{e:#x} ({e - s}B): {hot} samples = {100 * hot / total:.1f}% of all")
    log(f"   all dump regions together: {100 * sum(in_region.values()) / total:.1f}% of all samples")
    for ip, c in ips.most_common(15):
        where = next((f"+{ip - rs:#x} in {rs:#x}" for rs, re_ in regs if rs <= ip < re_), "outside dumps")
        log(f"   ip {ip:#x} {100 * c / total:5.1f}%  {where}")
    binf = os.path.join(out_dir, f"{name}-hot.bin")
    with open(binf, "wb") as f:
        f.write(bytes(mem[a] for a in range(s, e)))
    rc, dis = sh(["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
                  f"--adjust-vma={s:#x}", binf])
    rows = []
    for line in dis.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\t", line)
        if m:
            rows.append((ips.get(int(m.group(1), 16), 0), line))
    with open(os.path.join(out_dir, f"{name}-hot.dis.txt"), "w") as f:
        f.write("\n".join(f"{c:6d} {l}" for c, l in rows))
    if not rows:
        log(f"   objdump rc={rc}: {dis[:300]}")
        return
    width = 120
    best = max(range(max(1, len(rows) - width + 1)),
               key=lambda k: sum(c for c, _ in rows[k:k + width]))
    for c, l in rows[best:best + width]:
        log(f"   {c:6d} {l}")


def perf(runners, out_dir):
    section("4. perf")
    if sh(["perf", "stat", "--", "true"])[0] != 0:
        log("perf unavailable")
        return
    path = "pyre/bench/float_loop.py"
    trip = int(re.search(r"<\s*(\d+)", open(path).read()).group(1))
    log(f"float_loop trip count {trip}")
    for name, argv, env in runners:
        run_env = env if env is not None else dict(os.environ)
        rc, out = sh(["perf", "stat", "-x,", "-e", "cycles,instructions,branches,branch-misses", "--"]
                     + argv + [path], env=run_env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        vals = {}
        for line in out.splitlines():
            parts = line.split(",")
            if len(parts) > 2 and parts[0].strip().isdigit():
                vals[parts[2]] = int(parts[0])
        cyc = vals.get("cycles") or next((v for k, v in vals.items() if k.startswith("cycles")), 0)
        ins = vals.get("instructions") or next((v for k, v in vals.items() if k.startswith("instructions")), 0)
        ipc = f"{ins / cyc:.2f}" if cyc else "-"
        log(f"{name}: rc={rc} {vals} IPC={ipc} cycles/iter={cyc / trip:.2f} instr/iter={ins / trip:.2f}")
    for name, argv, env in runners:
        data = os.path.join(out_dir, f"{name}.perf.data")
        if name == "pyre":
            dump = os.path.join(out_dir, "pyre-perf-dump.txt")
            with open(dump, "wb") as f:
                sh(["perf", "record", "-F", "4999", "-o", data, "--"] + argv + [path],
                   env={**env, "MAJIT_DUMP": "1"}, stdout=subprocess.DEVNULL, stderr=f)
            mem = parse_pyre_dump(open(dump, encoding="utf-8", errors="replace").read())
        else:
            plog = os.path.join(out_dir, f"{name}-perf.pypylog")
            sh(["perf", "record", "-F", "4999", "-o", data, "--"] + argv + [path],
               env={**os.environ, "PYPYLOG": f"jit-backend-dump:{plog}"}, stdout=subprocess.DEVNULL)
            mem = parse_pypy_dump(open(plog, encoding="utf-8", errors="replace").read()) if os.path.exists(plog) else {}
        rc, ipout = sh(["perf", "script", "-F", "ip", "-i", data], stderr=subprocess.DEVNULL)
        ips = collections.Counter()
        for line in ipout.split():
            try:
                ips[int(line, 16)] += 1
            except ValueError:
                pass
        annotate(name, mem, ips, out_dir)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pypy", action="append", default=[])
    default_pyre = "target/release/pyre-dynasm" + (".exe" if sys.platform == "win32" else "")
    ap.add_argument("--pyre", default=default_pyre)
    ap.add_argument("--benches", default="float_loop,int_loop,nested_loop,fib_loop,spectral_norm,inline_helper")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--parse-only", help=argparse.SUPPRESS)
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    if args.parse_only:
        text = open(args.parse_only, encoding="utf-8", errors="replace").read()
        mem = parse_pyre_dump(text) if "[dynasm]" in text else parse_pypy_dump(text)
        for s, e in regions(mem):
            log(f"region {s:#x}..{e:#x} {e - s}B")
        return
    open(os.path.join(OUT, "summary.txt"), "w").close()
    runners = [("pyre", [args.pyre], check.pyre_env())]
    for spec in args.pypy:
        name, _, exe = spec.partition("=")
        runners.append((name, [exe], None))
    system_info(runners)
    timing(runners, args.benches.split(","), args.reps)
    code_dumps(runners, OUT)
    if args.perf:
        perf(runners, OUT)


if __name__ == "__main__":
    main()
