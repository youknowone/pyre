#!/usr/bin/env python3
"""Census of the frame-level FOR_ITER gate's declines over the bench corpus.

`eval.rs`'s FOR_ITER gate keeps a whole frame out of the tracer when any `for`
body holds an opcode outside `for_iter_body_op_is_jit_safe`'s allow-list, when
the loop is `finally`-duplicated, or when the frame has a re-raising `except E
as name:` handler.  The gate documents itself as debt —
`for_iter_bodies_all_jit_safe` opens with "This whole gate is a conservative
adaptation, not an upstream mechanism" — so the question that governs retiring
it is *which* predicate actually costs coverage, and on which frames.

This script answers that by running every `pyre/bench` and `pyre/bench/synth`
fixture under `PYRE_FOR_ITER_GATE_DIAG=1` and aggregating the two diagnostic
lines the gate emits:

    [for-iter-gate-decline] code=… source=… predicate=…
    [for-iter-gate-opcode]  code=… source=… for_iter_pc=… body_pc=… opcode=…

Counting **frames**, not events, is the load-bearing choice.  A declined helper
on a hot path emits hundreds of events and a declined module body emits one, so
an event count ranks by call frequency rather than by how much surface the gate
withdraws.  Both are reported; the frame count is what the work-list is ordered
by.

Frames are also split into the fixture's own code and `lib-python/`.  Every run
pays the stdlib declines during import, so pooling them buries the per-fixture
signal under a constant — and the stdlib side is where the two frame-level
predicates turn out to live at all (`re/_parser._parse`, `enum.EnumType.__new__`,
`pickle._Pickler.save_tuple`), which no synthetic fixture reaches.

Note that the two frame-level predicates print only as of the commit that made
`PYRE_FOR_ITER_GATE_DIAG` cover all four decline paths; they leave through
`unsupported_jit_shape`'s frame-shape arm, which used to return before the
`FrameGate::ForIter/…` print.  A census taken with an older binary silently reads
as if they never fire.

Usage, from the repo root, against a release build:

    python3 pyre/scripts/foriter-census.py --binary ./target/release/pyre-dynasm

A fixture that times out or exits non-zero is reported in the run-status line
rather than dropped silently, so a partial corpus cannot read as a clean one.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import os
import pathlib
import re
import subprocess
import sys

DECLINE_RE = re.compile(
    r"\[for-iter-gate-decline\] code=(?P<code>\S+) source=(?P<source>\S+) predicate=(?P<pred>\S+)"
)
OPCODE_RE = re.compile(
    r"\[for-iter-gate-opcode\] code=(?P<code>\S+) source=(?P<source>\S+) "
    r"for_iter_pc=(?P<fip>\d+) body_pc=(?P<bp>\d+) opcode=(?P<op>.+)$"
)
# `opcode={body_instr:?}` prints the full Debug form (`ImportName { i: Oparg(3)
# }`); the variant name alone is what ranks.
VARIANT_RE = re.compile(r"^([A-Za-z0-9_]+)")


def run_one(binary: str, fixture: pathlib.Path, timeout: int):
    env = dict(os.environ)
    env["PYRE_FOR_ITER_GATE_DIAG"] = "1"
    try:
        proc = subprocess.run(
            [binary, str(fixture)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return fixture, "timeout", [], []
    except OSError as exc:
        return fixture, f"oserror:{exc}", [], []
    status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
    declines, opcodes = [], []
    for line in proc.stderr.splitlines():
        m = DECLINE_RE.search(line)
        if m:
            declines.append((m["code"], m["source"], m["pred"]))
            continue
        m = OPCODE_RE.search(line)
        if m:
            variant = VARIANT_RE.match(m["op"].strip())
            opcodes.append(
                (
                    m["code"],
                    m["source"],
                    int(m["fip"]),
                    int(m["bp"]),
                    variant.group(1) if variant else m["op"].strip(),
                )
            )
    return fixture, status, declines, opcodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./target/release/pyre-dynasm")
    ap.add_argument("--root", default="pyre/bench")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    fixtures = sorted(root.glob("*.py")) + sorted((root / "synth").glob("*.py"))
    if not fixtures:
        print(f"no fixtures under {root}", file=sys.stderr)
        return 1

    def bucket(source: str) -> str:
        if "/lib-python/" in source or "/lib_pypy/" in source:
            return "stdlib"
        return "fixture"

    per_pred = collections.Counter()
    per_opcode = collections.Counter()
    # Distinct by (source, qualname): a helper declined in many fixtures is one
    # frame of withdrawn surface, not one per run.
    frames_by_pred = collections.defaultdict(set)
    frames_by_opcode = collections.defaultdict(set)
    fixtures_with_decline = set()
    statuses = collections.Counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(run_one, args.binary, f, args.timeout) for f in fixtures]
        for fut in concurrent.futures.as_completed(futs):
            fixture, status, declines, opcodes = fut.result()
            statuses[status] += 1
            if declines or opcodes:
                fixtures_with_decline.add(fixture.name)
            for code, source, pred in declines:
                per_pred[(bucket(source), pred)] += 1
                frames_by_pred[(bucket(source), pred)].add((source, code))
            for code, source, _fip, _bp, op in opcodes:
                per_opcode[(bucket(source), op)] += 1
                frames_by_opcode[(bucket(source), op)].add((source, code))

    print("# FOR_ITER gate decline census\n")
    print(f"- binary: `{args.binary}`")
    print(f"- corpus: {len(fixtures)} fixtures under `{args.root}` (+ `synth/`)")
    print("- run status: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    print(f"- fixtures emitting at least one decline: {len(fixtures_with_decline)}\n")

    for buck in ("fixture", "stdlib"):
        print(f"## [{buck}] whole-frame denials, by predicate\n")
        rows = [(k[1], n) for k, n in per_pred.most_common() if k[0] == buck]
        if rows:
            print("| predicate | events | distinct frames |")
            print("|---|---:|---:|")
            for pred, n in rows:
                print(f"| `{pred}` | {n} | {len(frames_by_pred[(buck, pred)])} |")
        else:
            print("_none_")
        print()

        print(f"## [{buck}] body opcodes that caused a denial\n")
        rows = [(k[1], n) for k, n in per_opcode.most_common() if k[0] == buck]
        if rows:
            print("| opcode | events | distinct frames |")
            print("|---|---:|---:|")
            for op, n in rows:
                print(f"| `{op}` | {n} | {len(frames_by_opcode[(buck, op)])} |")
        else:
            print("_none_")
        print()

    print("## Distinct declined frames\n")
    for (buck, pred), frames in sorted(
        frames_by_pred.items(), key=lambda kv: (kv[0][0], -len(kv[1]))
    ):
        print(f"### [{buck}] `{pred}` ({len(frames)})\n")
        for source, code in sorted(frames):
            print(f"- `{code}` — `{source}`")
        print()
    for (buck, op), frames in sorted(
        frames_by_opcode.items(), key=lambda kv: (kv[0][0], -len(kv[1]))
    ):
        print(f"### [{buck}] `{op}` ({len(frames)})\n")
        for source, code in sorted(frames):
            print(f"- `{code}` — `{source}`")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
