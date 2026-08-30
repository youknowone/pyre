#!/usr/bin/env python3
"""Hold the hand-written GC root brackets to what the analysis can prove.

`rpython/memory/gctransform/` inserts the shadow-stack bracket automatically:
`framework.py` brackets every operation that can reach the collector,
`shadowstack.py` emits the `gc_push_roots` / `gc_pop_roots` pair, and
`expand_pop_roots` turns the pop into one `gc_restore_root` per variable.
The graph-wide framework marker insertion and `shadowcolor` stages are now
automatic. Native interpreter paths are also compiled directly by rustc, so
they retain their existing source `push_roots` brackets and this gate stops
those brackets being taken on trust. See the module docs on
`majit-translate/src/memory/gctransform/mod.rs` for that current port boundary.

Two of the reported numbers are invariants at zero and are held there.  The
rest are a backlog: they are ratcheted, so a change may pay them down but not
add to them.

The baseline holds one entry per platform.  The scan reads an artefact built
from this platform's sources, and the interpreter's `cfg` arms differ across
them, so the counts do too -- a baseline written from one platform cannot be
satisfied from another, and `--update` rewrites only the entry it measured.

The ratchet is read against the base the baseline was taken on.  The backlog
counts every unbracketed call in the artefact, not this branch's share of
them, so a base that has moved brings code the baseline never saw into the
same number a regression would land in.  A rise measured over a moved base is
therefore reported and not failed; the invariants are held either way.

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


def platform_key() -> str:
    """The name this run's numbers are recorded under.

    The scan reads an artefact extracted from this platform's build, and the
    interpreter's `cfg` arms differ across them -- a Linux artefact carries
    calls a macOS one does not.  The counts are therefore not one number but
    one per platform, and a baseline written from one of them cannot be
    satisfied from another: a macOS `--update` would leave the Linux gate
    permanently red by exactly the difference between the two.
    """
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    return sys.platform


def merge_base() -> str:
    """The upstream commit this branch is measured against.

    The numbers below are a ratchet over a moving base.  A rebase brings
    interpreter code the baseline never saw, and its unbracketed calls land in
    this count exactly like a regression would -- so record what the baseline
    was taken against, and say when that has moved rather than let the reader
    infer that the rise is theirs.
    """
    # A shallow CI checkout is grafted: it holds no `main` ref, and the
    # merge commit's parent list is truncated away, so nothing in the
    # repository can name the base.  The workflow knows it and passes it in.
    supplied = os.environ.get("PYRE_GC_GATE_BASE", "").strip()
    if supplied:
        return supplied
    for base in ("origin/main", "upstream/main"):
        proc = subprocess.run(["git", "merge-base", base, "HEAD"], cwd=ROOT,
                              capture_output=True, encoding="utf-8")
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    # A pull-request checkout holds no `main` ref at all: the action fetches
    # `refs/pull/N/merge` and nothing else, so every `merge-base` above fails
    # and the base reads as unknown -- which is precisely the run that most
    # needs it, since that merge commit carries whatever `main` gained since
    # the baseline.  Its first parent *is* the base tip, and naming it needs
    # only the commit object already in hand.
    proc = subprocess.run(["git", "rev-list", "--parents", "-n", "1", "HEAD"],
                          cwd=ROOT, capture_output=True, encoding="utf-8")
    if proc.returncode == 0:
        parts = proc.stdout.split()
        if len(parts) == 3:
            return parts[1]
    return ""


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
    got["base"] = merge_base()
    key = platform_key()

    recorded = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {}

    if args.update:
        # Only this platform's entry: the others were measured on artefacts
        # this run never saw and are not ours to rewrite.
        recorded[key] = got
        BASELINE.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"wrote {BASELINE.relative_to(ROOT)} [{key}]")
        for k in sorted(got):
            print(f"  {k}: {got[k]}")
        return 0

    if key not in recorded:
        sys.exit(
            f"error: {BASELINE.relative_to(ROOT)} records no `{key}` entry "
            f"(it has: {', '.join(sorted(recorded)) or 'nothing'}).\n"
            "  Seed it from a run on this platform: "
            "python3 scripts/check-gc-root-brackets.py --update"
        )
    want = recorded[key]

    # Printed whichever way this ends, so a reader can see what was measured
    # rather than infer it from silence.
    print(f"gc root bracket gate — measured [{key}]:")
    for k in sorted(got):
        base = want.get(k, "—")
        mark = "" if got[k] == base else f"   (baseline {base})"
        print(f"  {k:34} {got[k]}{mark}")

    rebased = bool(got["base"]) and got["base"] != want.get("base")
    if rebased:
        print(
            f"\nNOTE: the baseline was taken against {want.get('base', '(unrecorded)')[:12]}"
            f" and this run sits on {got['base'][:12]}. Interpreter code the"
            f" baseline never saw is in this count; attribute a rise before"
            f" paying it down."
        )

    # A rise measured over a base the baseline never saw is not this branch's
    # to answer for: the backlog counts every unbracketed call in the artefact,
    # so interpreter code merged into the base since lands in it whole.  A pull
    # request is measured on its merge commit, so this is the ordinary case for
    # any branch whose base has moved, and failing it there accuses the branch
    # of a rise it did not cause.  The invariants are still held: those are
    # zero for the whole artefact whoever wrote the code.
    bad = []
    unattributed = []
    del got["base"]
    for k in INVARIANT_ZERO:
        if got[k] != 0:
            bad.append(f"{k} is {got[k]}, and this one is held at zero: a live "
                       f"pointer addressed as a relocatable object across an "
                       f"unbracketed collecting call is a use-after-move, not "
                       f"a backlog entry.")
    for k in RATCHET:
        if k in want and got[k] > want[k]:
            rise = (f"{k} rose {want[k]} -> {got[k]}. Bracket the new call "
                    f"with `pyre_object::with_roots!`, or pay the baseline "
                    f"down and rerun with --update if the rise is real and "
                    f"intended.")
            (unattributed if rebased else bad).append(rise)
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
    if unattributed:
        print("\nWARN — a raised backlog this run cannot attribute:")
        for u in unattributed:
            print(f"  - {u}")
        print("  Rebase onto the recorded base and rerun to attribute these, "
              "or rebaseline from a run that sits on it.")
        return 0
    print("\nOK — invariants at zero, backlog not raised.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
