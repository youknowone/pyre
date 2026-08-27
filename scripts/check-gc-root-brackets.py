#!/usr/bin/env python3
"""Hold the hand-written GC root brackets to what the analysis can prove.

`rpython/memory/gctransform/` inserts the shadow-stack bracket automatically:
`framework.py` brackets every operation that can reach the collector,
`shadowstack.py` emits the `gc_push_roots` / `gc_pop_roots` pair, and
`expand_pop_roots` turns the pop into one `gc_restore_root` per variable.
pyre's interpreter is compiled by rustc, so the bracket is written by hand and
this gate is what stops it being taken on trust -- see the module docs on
`majit-translate/src/memory/gctransform/mod.rs` for why the insertion half is
out of reach for this pipeline.

Two of the reported numbers are invariants at zero and are held there.  The
rest are a backlog: they are ratcheted, so a change may pay them down but not
add to them.

Run the analysis and compare:

    cargo build -p majit-translate --release --example gc-root-reachability
    python3 scripts/check-gc-root-brackets.py

`--update` rewrites the baseline from the current run, for a change that pays
the backlog down.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "majit" / "gc-root-brackets.baseline.json"
EXAMPLE = ROOT / "target" / "release" / "examples" / "gc-root-reachability"

# The donor set is not a matter of taste: the example's own header records what
# each donor moves, and leaving `pyre-object.ullbc` out fails quietly rather
# than loudly -- the collecting-allocation seed lives there and scores zero
# against the interpreter artefact alone.
SUBJECT = "build/llbc/pyre-interpreter.ullbc"
DONORS = [
    "build/llbc/pyre-jit.ullbc",
    "build/llbc/pyre-object.ullbc",
    "build/llbc/majit-rlib.ullbc",
]

# (key, regex, how many groups to keep).  Every one of these must match exactly
# once, in this order: `tier 1` is printed twice, once for the main scan and
# once for the frame scan, and telling them apart is positional.
PATTERNS = [
    ("unmatched_seeds",
     r"collecting-alloc seeds:.*?UNMATCHED patterns: \[(?P<v>.*?)\]"),
    ("brackets_reaching_no_collection",
     r"cannot reach any collection\s*: (?P<v>\d+)"),
    ("unbracketed_calls",
     r"unbracketed calls that can collect with a live PyObjectRef: (?P<v>\d+) in (?P<w>\d+) fn"),
    ("tier1_calls",
     r"tier 1 \(callee IS a dispatch seed\): (?P<v>\d+) call\(s\) in (?P<w>\d+) fn"),
    ("tier15_calls",
     r"tier 1\.5 \(live ptr later addressed as list/dict\): (?P<v>\d+) call\(s\) in (?P<w>\d+) fn"),
    ("frames_across_collecting",
     r"frame carried across a call that can collect: (?P<v>\d+) in (?P<w>\d+) fn"),
    ("frame_tier1_calls",
     r"tier 1 \(callee IS a dispatch seed\): (?P<v>\d+) call\(s\) in (?P<w>\d+) fn"),
]

# Held at zero rather than ratcheted.  `tier 1.5` is a live pointer later
# addressed as a list or dict -- the two kinds a minor collection relocates, so
# a stale one is dereferenced as a corpse rather than merely stored.  A frame
# carried across a collecting call whose callee is a dispatch seed is the same
# hazard for the frame itself.
INVARIANT_ZERO = ("tier15_calls", "frame_tier1_calls")

# Ratcheted: may fall, may not rise.
RATCHET = (
    "unbracketed_calls",
    "tier1_calls",
    "frames_across_collecting",
    "brackets_reaching_no_collection",
)


def run_analysis() -> str:
    if not EXAMPLE.exists():
        sys.exit(
            f"error: {EXAMPLE.relative_to(ROOT)} is not built.\n"
            "  cargo build -p majit-translate --release "
            "--example gc-root-reachability"
        )
    missing = [p for p in [SUBJECT, *DONORS] if not (ROOT / p).is_file()]
    if missing:
        sys.exit(
            "error: LLBC artefacts missing: " + ", ".join(missing) + "\n"
            "  python3 scripts/extract-llbc.py majit-rlib pyre-object "
            "pyre-interpreter pyre-jit"
        )
    env = dict(os.environ, GC_JOIN_WITH=",".join(DONORS))
    proc = subprocess.run([str(EXAMPLE), SUBJECT], cwd=ROOT, env=env,
                          capture_output=True, encoding="utf-8",
                          errors="replace")
    if proc.returncode != 0:
        sys.exit(f"error: analysis exited {proc.returncode}\n{proc.stderr}")
    return proc.stdout


def parse(report: str) -> dict:
    """Read the numbers, and refuse to report a clean run over an empty read.

    A gate whose pass is indistinguishable from a gate that matched nothing is
    a gate nobody can trust, so a pattern that does not appear where it is
    expected is an error rather than a missing key.
    """
    got: dict = {}
    pos = 0
    for key, pattern in PATTERNS:
        m = re.compile(pattern, re.S).search(report, pos)
        if m is None:
            sys.exit(
                f"error: the analysis report has no `{key}` line after "
                f"offset {pos}. The report shape changed; this gate reads it "
                f"positionally and cannot tell a zero from an absence.\n"
                f"--- report ---\n{report}"
            )
        pos = m.end()
        if key == "unmatched_seeds":
            names = [s.strip().strip('"') for s in m.group("v").split(",")]
            got[key] = sorted(n for n in names if n)
        else:
            got[key] = int(m.group("v"))
            if "w" in m.groupdict() and m.group("w") is not None:
                got[key + "_fns"] = int(m.group("w"))
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true",
                    help="rewrite the baseline from this run")
    args = ap.parse_args()

    got = parse(run_analysis())

    if args.update:
        BASELINE.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        print(f"wrote {BASELINE.relative_to(ROOT)}")
        for k in sorted(got):
            print(f"  {k}: {got[k]}")
        return 0

    if not BASELINE.is_file():
        sys.exit(f"error: no baseline at {BASELINE.relative_to(ROOT)}; "
                 "seed it with --update")
    want = json.loads(BASELINE.read_text())

    # Printed whichever way this ends, so a reader can see what was measured
    # rather than infer it from silence.
    print("gc root bracket gate — measured:")
    for k in sorted(got):
        base = want.get(k, "—")
        mark = "" if got[k] == base else f"   (baseline {base})"
        print(f"  {k:34} {got[k]}{mark}")

    bad = []
    for k in INVARIANT_ZERO:
        if got[k] != 0:
            bad.append(f"{k} is {got[k]}, and this one is held at zero: a live "
                       f"pointer addressed as a relocatable object across an "
                       f"unbracketed collecting call is a use-after-move, not "
                       f"a backlog entry.")
    for k in RATCHET:
        if k in want and got[k] > want[k]:
            bad.append(f"{k} rose {want[k]} -> {got[k]}. Bracket the new call "
                       f"with `pyre_object::with_roots!`, or pay the baseline "
                       f"down and rerun with --update if the rise is real and "
                       f"intended.")
    if "unmatched_seeds" in want and got["unmatched_seeds"] != want["unmatched_seeds"]:
        bad.append(
            f"the unmatched seed set changed: {want['unmatched_seeds']} -> "
            f"{got['unmatched_seeds']}. A seed that matches nothing empties "
            f"half the closure silently, so this is checked rather than "
            f"trusted."
        )

    if bad:
        print("\nFAIL")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("\nOK — invariants at zero, backlog not raised.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
