#!/usr/bin/env python3
"""Refuse to write over a file another agent is holding uncommitted.

Several agents share one worktree. `git commit --only <paths>` protects the
*commit* from capturing a peer's staged work; nothing protected the *write*.
Those are different hazards, and the write is the worse one: working-tree-only
content has never entered the object store, so after an overwrite there is no
dangling blob, nothing staged, and no recovery route -- every route git offers
operates on objects.

Three properties are load-bearing and all three are why this is a script rather
than a rule someone is asked to remember:

1. PER-FILE.  A batch check is a snapshot of the wrong instant for files 2..N.
2. ADJACENT.  The check runs in the same loop iteration as the write it guards,
   with nothing in between.  A status read is a snapshot, not a lock; on a
   shared tree it expires at the next keystroke by anyone.
3. FAIL CLOSED.  Refuse and exit non-zero.  A warning that proceeds is a
   diagnostic, and a diagnostic printed beside an action is not a guard --
   a guard is a branch that can refuse.

Modes:

    guard  PATH...              refuse if any PATH is dirty; run adjacent to a
                                write this script does not perform itself
    copy   --from RIG PATH...   guarded copy of RIG/PATH -> PATH, re-checking
                                immediately before each individual write
    window                      refuse unless the whole tree is clean; the
                                precondition for taking an LLBC extract window

Exit codes:  0 allowed / 1 refused / 2 the guard could not decide.
Code 2 is distinct on purpose -- "I could not tell" must never be spelled the
same as "clean".

NON-GOAL: THIS CANNOT SEE A TRANSIENT WRITE, AND A CLEAN `status` DOES NOT
CLEAR YOU DURING SOMEONE ELSE'S BUILD.

Every check here reads a *state*.  The worst shared-tree hazard is an
*interval*: a file edited and then reverted or committed leaves `git status`
clean and `git diff` empty afterwards, so no reading taken before or after can
detect that it was dirty in between.

That is not hypothetical.  An LLBC extract hashed
`majit-macros/.../codegen_state.rs` while it was uncommitted, and the file was
clean again minutes later.  The artefact was **stamped** and reads
`fingerprint matches the tree` -- over bytes that at that moment existed in no
commit.  It only still matches because the edit later landed; had the author
reverted instead, the stamp would attest bytes existing nowhere, `git log`
would show nothing and `git status` would be clean.  Note which way that fails:
a *commit* during a build trips the extractor's loud refusal, while an
*uncommitted* edit trips nothing and produces a quiet success.  The artefact
that refuses is the safe one.

So this script protects a write you are about to make.  It cannot certify a
window for someone else's long-running build, and running it is not a defence
against having disturbed one.  A build window is a **commitment from every
writer** ("nothing queued, I have stopped"), collected before the window opens
-- not a reading the builder takes.  No sampling rate turns "clean at instant
T" into "quiet for the next eight minutes".
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ALLOWED, REFUSED, UNDECIDED = 0, 1, 2


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command and hand back the whole result.

    Never pipe a command whose exit code is the answer: a pipeline reports the
    *last* stage, so `git ... | head` returns head's success and hides git's
    `fatal:`.  Reading .returncode off the process itself is the only form that
    cannot be laundered.
    """
    return subprocess.run(args, capture_output=True, text=True)


def repo_root() -> Path:
    proc = _run(["git", "rev-parse", "--show-toplevel"])
    if proc.returncode != 0:
        sys.exit(die(f"not inside a git worktree: {proc.stderr.strip()}"))
    return Path(proc.stdout.strip())


def die(msg: str) -> int:
    print(f"UNDECIDED: {msg}", file=sys.stderr)
    return UNDECIDED


def status_of(root: Path, rel: str) -> tuple[str, str | None]:
    """Return (verdict, porcelain_code) for one path.

    verdict is "clean", "dirty", "absent", or "undecided".

    `--no-optional-locks` keeps this read from refreshing `.git/index`.  Ask of
    any guard what it writes: a precondition check that mutates the thing it
    certifies is another writer wearing a guard's name.
    """
    proc = _run(
        [
            "git",
            "--no-optional-locks",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "-z",
            "--",
            rel,
        ]
    )
    if proc.returncode != 0:
        return "undecided", proc.stderr.strip()

    entries = [e for e in proc.stdout.split("\0") if e]
    if not entries:
        # Empty output has two causes: the path is clean, or the pathspec
        # matched nothing.  They are NOT the same, and reading the second as
        # the first is how a guard fails open.  Split them on disk.
        return ("clean", None) if (root / rel).exists() else ("absent", None)

    return "dirty", entries[0][:2]


def check(root: Path, rel: str, *, absent_ok: bool) -> int:
    verdict, detail = status_of(root, rel)

    if verdict == "undecided":
        return die(f"git status failed for {rel}: {detail}")
    if verdict == "dirty":
        print(f"REFUSING: {rel} is dirty [{detail}] -- another agent is "
              f"holding it uncommitted", file=sys.stderr)
        return REFUSED
    if verdict == "absent":
        if absent_ok:
            print(f"  new     {rel}")
            return ALLOWED
        print(f"REFUSING: {rel} matches nothing in {root} -- a pathspec that "
              f"matches nothing reads as clean, so this is refused rather "
              f"than assumed safe", file=sys.stderr)
        return REFUSED

    print(f"  clean   {rel}")
    return ALLOWED


def mode_guard(root: Path, paths: list[str]) -> int:
    worst = ALLOWED
    for rel in paths:
        rc = check(root, rel, absent_ok=False)
        worst = max(worst, rc)
    return worst


def mode_copy(root: Path, rig: Path, paths: list[str]) -> int:
    if not rig.is_dir():
        return die(f"rig {rig} is not a directory")

    # Verify every source up front.  A mistyped or unsplit path shows up here,
    # as a missing SOURCE that fails closed -- not downstream as an empty
    # status read on the destination, which would fail open.
    missing = [p for p in paths if not (rig / p).is_file()]
    if missing:
        for p in missing:
            print(f"UNDECIDED: no source at {rig / p}", file=sys.stderr)
        return UNDECIDED

    for rel in paths:
        # Adjacent by construction: the check and the write it guards are the
        # same iteration, and re-reading per file is what makes the snapshot
        # cover the instant it is used.
        rc = check(root, rel, absent_ok=True)
        if rc != ALLOWED:
            print(f"STOPPED before writing {rel}; "
                  f"{len(paths) - paths.index(rel)} file(s) not copied",
                  file=sys.stderr)
            return rc
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rig / rel, dst)
        print(f"  copied  {rel}")
    return ALLOWED


def mode_window(root: Path) -> int:
    proc = _run(["git", "--no-optional-locks", "-C", str(root),
                 "status", "--porcelain", "-z"])
    if proc.returncode != 0:
        return die(f"git status failed: {proc.stderr.strip()}")

    entries = [e for e in proc.stdout.split("\0") if e]
    tracked = [e for e in entries if not e.startswith("??")]
    if not tracked:
        print("tree clean -- but clean at this instant is not a lock; the "
              "window is a commitment from every writer, not a reading")
        return ALLOWED

    print(f"REFUSING: {len(tracked)} tracked path(s) dirty", file=sys.stderr)
    for e in tracked:
        print(f"  {e}", file=sys.stderr)
    print("\nA stopped writer with uncommitted work looks exactly like a "
          "running one from here, and is worse: the running writer finishes "
          "and commits, while the stopped one leaves a state only they can "
          "resolve. Extracting now would stamp a fingerprint over bytes that "
          "exist in no commit.", file=sys.stderr)
    return REFUSED


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("guard", help="refuse if any PATH is dirty")
    g.add_argument("paths", nargs="+", metavar="PATH")

    c = sub.add_parser("copy", help="guarded per-file copy from a rig")
    c.add_argument("--from", dest="rig", required=True, metavar="RIG")
    c.add_argument("paths", nargs="+", metavar="PATH")

    sub.add_parser("window", help="refuse unless the whole tree is clean")

    args = ap.parse_args()
    root = repo_root()

    if args.mode == "guard":
        return mode_guard(root, args.paths)
    if args.mode == "copy":
        return mode_copy(root, Path(args.rig).resolve(), args.paths)
    return mode_window(root)


if __name__ == "__main__":
    sys.exit(main())
