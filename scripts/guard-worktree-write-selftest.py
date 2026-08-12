#!/usr/bin/env python3
"""Self-test for `guard-worktree-write.py`, with four outcomes rather than two.

The guard it grades exists because a shared worktree is written by several
agents at once.  The first run of this matrix was destroyed by exactly that:
it used a peer's dirty files as the "dirty" specimen, the peer committed
mid-run, and three arms reported FAIL naming the guard.  The guard was right
and the fixture was gone.

Two lessons are built in here rather than written down:

1. A FIXTURE IS NOT A CONSTANT.  Every arm states the precondition its
   assertion needs and that precondition is re-read AROUND the run -- before
   and after -- so a fixture that moved is reported as FIXTURE-LOST, its own
   outcome, instead of being folded into FAIL.  A missing precondition
   reported as a finding about the subject is worse than a crash, because it
   is actionable-looking and sends the next reader to the wrong component.

2. COMPARE THE EXIT CODE FIRST.  The original harness compared file hashes
   without consulting rc and printed ticks for a program that never ran
   (rc=127).  A content comparison alone cannot tell "correctly did nothing"
   from "never executed", so rc is checked before any content claim and a
   subject that could not run at all is UNDECIDED, not FAIL.

FIXTURE-LOST and UNDECIDED are kept apart on purpose: a fixture that moved
and a subject that never executed are different failures, and merging them
would repeat the merge this file exists to fix.

Everything happens in a throwaway repository this script creates and removes.
It never writes into the repository under test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GUARD = Path(__file__).resolve().parent / "guard-worktree-write.py"

PASS, FAIL = "PASS", "FAIL"
FIXTURE_LOST = "FIXTURE-LOST"   # the precondition moved; says nothing about the guard
UNDECIDED = "UNDECIDED"         # the subject never executed; says nothing either
CONTROL_INERT = "CONTROL-INERT"  # a control did not fire ⇒ the arm it backs is unmeasured
NOT_A_RESULT = (FIXTURE_LOST, UNDECIDED, CONTROL_INERT)
COULD_NOT_RUN = 127

results: list[tuple[str, str, str]] = []


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def porcelain(repo: Path, rel: str) -> str:
    """The fixture's own state, read fresh.  Never cached.

    STOP: The first version of this helper reproduced the exact defect the guard
    under test exists to fix.  `git status --porcelain -- <missing path>` exits
    **0 with empty output**, so keying "absent" on a non-zero return code never
    fired and a path that did not exist reported `clean`.  Empty output has
    three causes here -- clean, absent, unmatched -- and they must be split on
    disk, not inferred from an exit code that is 0 for all three.
    """
    proc = git(repo, "--no-optional-locks", "status", "--porcelain", "--", rel)
    if proc.returncode:
        return "undecided"
    if proc.stdout:
        return proc.stdout[:2]
    return "clean" if (repo / rel).exists() else "absent"


def sha(path: Path) -> str | None:
    return None if not path.is_file() else str(path.stat().st_size) + ":" + \
        str(hash(path.read_bytes()))


def run_guard(repo: Path, *args: str) -> int:
    return subprocess.run([sys.executable, str(GUARD), *args],
                          cwd=repo, capture_output=True, text=True).returncode


def arm(name: str, repo: Path, expect_rc: int, fixture: dict[str, str],
        call: list[str], content_claim=None) -> None:
    """One assertion.

    `fixture` maps a path to the porcelain code the assertion REQUIRES.  It is
    verified immediately before and immediately after the run, so a state that
    moved during the window is caught rather than assumed.
    """
    before = {p: porcelain(repo, p) for p in fixture}
    if before != fixture:
        results.append((name, FIXTURE_LOST, f"before: wanted {fixture}, saw {before}"))
        return

    snapshots = {p: sha(repo / p) for p in (content_claim or {})}
    rc = run_guard(repo, *call)

    if rc == COULD_NOT_RUN:
        results.append((name, UNDECIDED, "subject did not execute (rc=127)"))
        return

    # The fixture must have held for the whole window, not just at setup.
    # Arms that are SUPPOSED to change a file exempt that path.
    after = {p: porcelain(repo, p) for p in fixture
             if not (content_claim or {}).get(p) == "changed"}
    stale = {p: (fixture[p], after[p]) for p in after if after[p] != fixture[p]}
    if stale:
        results.append((name, FIXTURE_LOST,
                        f"moved during the run: {stale} -- says nothing about the guard"))
        return

    if rc != expect_rc:
        results.append((name, FAIL, f"expected rc={expect_rc}, got {rc}"))
        return

    for path, claim in (content_claim or {}).items():
        now = sha(repo / path)
        changed = now != snapshots[path]
        if claim == "unchanged" and changed:
            results.append((name, FAIL, f"{path} WAS WRITTEN despite rc={rc}"))
            return
        if claim == "changed" and not changed:
            results.append((name, FAIL, f"{path} was not written though rc={rc}"))
            return

    results.append((name, PASS, f"rc={rc}"))


def build_fixture(repo: Path, rig: Path) -> None:
    repo.mkdir(parents=True)
    rig.mkdir(parents=True)
    git(repo, "init", "-q", ".")
    git(repo, "config", "user.email", "selftest@local")
    git(repo, "config", "user.name", "selftest")
    for f in ("clean.txt", "dirty.txt", "staged.txt", "second.txt"):
        (repo / f).write_text("COMMITTED\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")

    (repo / "dirty.txt").write_text("PEER-EDIT\n")
    (repo / "staged.txt").write_text("PEER-STAGED\n")
    git(repo, "add", "staged.txt")
    (repo / "untracked.txt").write_text("PEER-NEW\n")

    # The rig content DIFFERS from every destination.  With identical bytes,
    # "unchanged after refusal" would pass even if the write had gone through.
    for f in ("clean.txt", "dirty.txt", "staged.txt", "untracked.txt",
              "second.txt", "brand-new.txt"):
        (rig / f).write_text("RIG-DIFFERENT\n")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="guard-selftest-"))
    repo, rig = tmp / "repo", tmp / "rig"
    try:
        build_fixture(repo, rig)
        R = str(rig)

        arm("guard: clean tracked", repo, 0, {"clean.txt": "clean"}, ["guard", "clean.txt"])
        arm("guard: modified ' M'", repo, 1, {"dirty.txt": " M"}, ["guard", "dirty.txt"])
        arm("guard: staged 'M '", repo, 1, {"staged.txt": "M "}, ["guard", "staged.txt"])
        arm("guard: untracked '??'", repo, 1, {"untracked.txt": "??"}, ["guard", "untracked.txt"])
        arm("guard: bogus pathspec", repo, 1, {}, ["guard", "a.txt b.txt"])
        arm("guard: batch, one dirty", repo, 1, {"clean.txt": "clean", "dirty.txt": " M"},
            ["guard", "clean.txt", "dirty.txt"])

        arm("copy: missing source", repo, 2, {}, ["copy", "--from", R, "nope.txt"])
        for f, code in (("dirty.txt", " M"), ("staged.txt", "M "), ("untracked.txt", "??")):
            arm(f"copy: onto {code!r} refuses and does not write", repo, 1, {f: code},
                ["copy", "--from", R, f], {f: "unchanged"})
        arm("copy: onto clean OVERWRITES (positive control)", repo, 0, {"clean.txt": "clean"},
            ["copy", "--from", R, "clean.txt"], {"clean.txt": "changed"})
        arm("copy: to a new path", repo, 0, {"brand-new.txt": "absent"},
            ["copy", "--from", R, "brand-new.txt"], {"brand-new.txt": "changed"})
        arm("copy: dirty FIRST stops the file after it", repo, 1,
            {"dirty.txt": " M", "second.txt": "clean"},
            ["copy", "--from", R, "dirty.txt", "second.txt"], {"second.txt": "unchanged"})

        arm("window: tracked dirt present", repo, 1, {"dirty.txt": " M"}, ["window"])
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "peer lands everything")
        arm("window: tree clean", repo, 0, {"dirty.txt": "clean"}, ["window"])

        index = repo / ".git" / "index"
        # git refreshes the index only when its stat cache is stale, so both
        # arms below are run from the same deliberately-staled state. Without
        # this the control cannot fire and the hygiene arm is unproven.
        def stale_the_stat_cache() -> None:
            os.utime(repo / "clean.txt", ns=(0, 0))

        stale_the_stat_cache()
        m0 = index.stat().st_mtime_ns
        for _ in range(50):
            run_guard(repo, "guard", "clean.txt")
        hygiene_held = index.stat().st_mtime_ns == m0
        results.append(("hygiene: 50 guard calls leave .git/index alone",
                        PASS if hygiene_held else FAIL, ""))

        # CONTROL for the arm above: `--no-optional-locks` is only load-bearing
        # if the UNFLAGGED form does rewrite the index. If this does not fire,
        # the hygiene arm measured nothing -- that is not a FAIL of the guard,
        # it is an inert experiment, and conflating the two would report a
        # missing measurement as a finding about the subject.
        stale_the_stat_cache()
        m1 = index.stat().st_mtime_ns
        git(repo, "status", "--porcelain")
        if index.stat().st_mtime_ns != m1:
            results.append(("control: plain `git status` DOES rewrite .git/index",
                            PASS, "so the hygiene arm above is a real measurement"))
        else:
            results.append(("control: plain `git status` DOES rewrite .git/index",
                            CONTROL_INERT,
                            "control did not fire ⇒ the hygiene arm proves nothing "
                            "about --no-optional-locks; it is unmeasured, not green"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    width = max(len(n) for n, _, _ in results)
    for name, verdict, detail in results:
        print(f"  {verdict:<12} {name:<{width}}  {detail}")

    tally = {v: sum(1 for _, x, _ in results if x == v)
             for v in (PASS, FAIL, *NOT_A_RESULT)}
    unmeasured = sum(tally[v] for v in NOT_A_RESULT)
    print(f"\n{tally[PASS]}/{len(results)} PASS, {tally[FAIL]} FAIL, "
          f"{tally[FIXTURE_LOST]} FIXTURE-LOST, {tally[UNDECIDED]} UNDECIDED, "
          f"{tally[CONTROL_INERT]} CONTROL-INERT")
    if unmeasured:
        print(f"WARNING: {unmeasured} arm(s) produced NO RESULT ABOUT THE GUARD -- a lost "
              "fixture, a subject that never ran, or a control that did not fire. "
              "Re-run; do not read them as reds, and do not read the arms they "
              "back as greens.")
    return 1 if tally[FAIL] or unmeasured else 0


if __name__ == "__main__":
    sys.exit(main())
