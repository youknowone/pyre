#!/usr/bin/env python3
"""Assert that every workspace crate reaching majit-metainterp selects a backend.

majit-metainterp has a mandatory-backend `compile_error!` (pyjitpl.rs).  The
workspace root depends on it with `default-features = false`, so metainterp's
own `default = ["dynasm"]` does not apply across that edge: a crate on the far
side must select a backend ITSELF, or `cargo test -p <crate>` builds ZERO test
targets.  Five crates were in that state simultaneously and nothing caught it.

WHY CI CANNOT SEE IT WITHOUT THIS TOOL
    CI runs `--all --no-default-features --features dynasm`, which supplies a
    backend workspace-wide and never issues the per-package form.  The defect
    is invisible to every green run.

WHY `cargo tree` AND NOT A BUILD
    This is a feature-RESOLUTION defect, and resolution happens before build
    scripts run.  Two properties follow, and the second is the one that matters:

      * it costs seconds instead of a cold rebuild of the world, and
      * it stays readable when the LLBC staleness gate is red -- which is the
        exact condition that hid five crates for a full session.  A build-based
        check cannot have that property, because the staleness gate lives in a
        build script and stops the build before any of this is decided.

    A build-based check also cannot be made cheap by passing
    `--no-default-features --features dynasm`: supplying a backend explicitly is
    precisely what masks the defect.

WHY THE PROBE HAS TWO STAGES, AND WHY STAGE 1 IS NOT OPTIONAL
    An empty stage-2 result has TWO causes and only one is a defect:

      1. the crate reaches majit-metainterp and selects no backend -- the bug;
      2. the crate does not reach majit-metainterp at all -- perfectly healthy.

    A one-stage probe reports them identically.  The first version of this check
    was one-stage, and `majit-rlib` is case 2: it would have demanded a backend
    feature on a crate that must not have one.  `cargo tree -i <pkg>` exits
    nonzero when <pkg> is not in the graph, so the scope question is answered by
    an exit code that was already there.

    WARNING: majit-rlib is worth keeping as the worked example: it depends on
    majit-macros, and majit-macros reaches majit-metainterp only through a
    `[dev-dependencies]` entry.  Dev-dependencies do not propagate to a
    dependent's build, so the path dies one hop in.  The NAME is present two
    hops up and the GRAPH EDGE is not.

WHY THE POPULATION IS THE DEPENDENCY GRAPH AND NOT THE MANIFEST TEXT
    Same reason, in the other direction.  Deriving the population by grepping
    manifests for "majit-metainterp" undercounts: `pyre-module` and
    `pyre-wasm-test` reach metainterp transitively and name it nowhere.  Those
    are exactly the crates that inherit a backend across a plain edge -- the
    case this rule exists to describe -- so the text predicate silently deletes
    the interesting middle and leaves a plausible number.

WHY THE CONTROLS ARE SYNTHESIZED RATHER THAN A HISTORICAL TREE
    The obvious control is the tree that predates the fix, and it was used once,
    by hand, to validate this probe (it reproduced the five failures exactly).
    It is the wrong thing to WIRE IN.

      * A branch-local sha does not survive a rebase.  The one used here was
        `eba7c58ad35^`, and it is now dead exactly as predicted -- no longer an
        ancestor of HEAD, while `git cat-file -t` still answers `commit`,
        because the object is merely dangling.  A test pinned to it therefore
        PASSES on the machine that wrote it and dies in CI or after `git gc`.
        That is worse than a citation that fails immediately.  WARNING: That sha is
        LEFT DEAD on purpose -- it is the evidence for this paragraph, not a
        citation to follow.  The fix it precedes is "majit,pyre: put a backend
        in the default of every crate that needs one to build", whose rebased
        successor was `76d4cda71b4` on 2026-08-11; find it by that subject.
      * More importantly, no commit in this repo's history contains a workspace
        member that is OUT of the population.  So a historical control cannot
        exercise case 2 above at all -- the very blind spot that made the first
        version of this check wrong would still be invisible.

    The three synthesized controls below cover the full truth table, cost
    milliseconds, and cannot rot.

WHAT THIS IS BLIND TO -- read this before quoting a clean run
    The name says "backend edge", which will be read as covering more than it
    does.  Against the four coupling axes recorded in task #79 (they do NOT
    nest -- axis 4's population is our own source files, not consumer edges):

      1. INVISIBLE TO A TRACKED-TREE CENSUS.  The population is WORKSPACE
         MEMBERS.  Two path-dependency consumers (`wasmi/`, `wasmi-majit-pr/`)
         are separate git repos nested in this tree, compiling against this
         majit, and `cargo metadata` on this workspace does not see them.  A
         clean run here says nothing about whether they select a backend.
      2. BREAKS WHEN WE EDIT MAJIT SOURCE.  Those same path-dep consumers are
         the risk set; a `rev = `-pinned consumer (cel-jit) is not, because the
         pin holds it at a frozen majit.  This check covers neither.
      3. DESYNCS WHEN A GATE STRING IS RENAMED.  Out of scope by construction,
         not by omission: this tool reads the FEATURE graph, and an env-var gate
         contract is a runtime string with no dependency edge to resolve.  No
         extension of this check reaches it.

      4. BUILD-BLOCKING ARTEFACT COUPLING, and it is not a manifest edge at
         all.  Editing a closure input under `majit/` or `pyre/` stales
         `build/llbc/*.ullbc` and HARD-FAILS the build of everything downstream
         of `pyre-jit-trace` (`error: LLBC STALE: … sources now hash to …`) --
         a stale artefact, not merely a missing one.  The coupling runs through
         an ARTEFACT DIRECTORY checked by a build script, so there is no edge
         for a graph query to traverse.
         ⇒ Its membership question has a real instrument, unlike axes 1-3:
           `python3 pyre/scripts/extract-llbc.py --list-inputs <crate>`, and
           `BASE_PATHSPECS` in that same file declares the cross-crate inputs
           in six lines.  The closure is exactly enumerable -- 728 files over
           15 library crates, `majit/examples/**` contributing ZERO.
           STOP: Ask it per FILE.  "`scripts/` is outside the closure" is false as
           a scope -- FOUR of the five `scripts/`-family files are inputs
           (`llbc_extract.py`, both `extract-llbc.py`, `install-charon.py`) --
           while "this file is not among the 728" is true as a predicate.
           THIS file is not among them; the directory it sits in is not what
           makes that so.
         WARNING: DETECTION EXISTS, PREVENTION DOES NOT.  Nothing stops anyone editing
           a closure member; it broke the build twice on 2026-08-10 and both
           times the remedy was a human calling a freeze.  This file is a
           MECHANISM -- it fails a build.  That is a PROCESS CONTROL.  A reader
           who trusts them equally is trusting the second one wrongly.

    ⭐ This is the same mechanism as `WHY cargo tree AND NOT A BUILD` above,
    read from the other side, and the pair is worth holding together: feature
    resolution happens BEFORE build scripts run, which is exactly why this tool
    stays readable when the staleness gate is red -- and exactly why it cannot
    see staleness coupling.  The property is one property.  You do not get to
    keep the robustness and also claim the coverage.

    WARNING: THE LIST IS A PARTITION, NOT A BACKLOG.  1 and 2 are gaps this tool could
    close by widening the population to the nested checkouts.  3 and 4 are not,
    and the difference misleads in both directions: filing them as backlog items
    spends a day discovering there is no edge to resolve, while reading "no
    extension of this check" as "nothing can check it" forecloses work that is
    merely a DIFFERENT instrument.  Both already have one named -- a text sweep
    for 3, `--list-inputs` for 4 -- and neither needs feature resolution, which
    is exactly why neither belongs in this file.

    So a green `backend-edge` job means: every workspace member that reaches
    majit-metainterp in its dependency graph selects a backend, checked across
    MANIFEST edges only.  It does not mean the backend edge is safe repo-wide,
    and it says nothing about LLBC closure coupling.

⭐ IT RESOLVES; IT NEVER COMPILES -- AND BOTH HALVES MATTER TO A READER
    `cargo tree` and `cargo metadata` resolve features and stop.  Build scripts
    never run, so `pyre-jit-trace`'s LLBC freshness check never fires and this
    leg needs no `PYRE_LLBC_STRICT=0`.  Measured 2026-08-10, both failure modes
    a CI runner can present, with the env explicitly unset:

      * STALE artefacts -- all four `build/llbc` stamps measurably stale at the
        instant of the run (the tree-wide `RC=101` state): 24/14/0, RC=0.
      * ABSENT artefacts -- `git archive HEAD` into a scratch tree, no
        `build/llbc` at all (`build/llbc` is untracked, so this IS a fresh
        checkout): 24/14/0, RC=0, and no `build/llbc` created afterwards,
        which is the direct evidence that no build script ran.

    STOP: THE COST OF THAT PROPERTY, and it is the half a green run hides: this leg
    CANNOT TELL YOU THE WORKSPACE COMPILES.  It is not a smoke test.  Reading a
    green `backend-edge` as "the tree builds" is the same over-read the
    `WHAT THIS IS BLIND TO` section exists to prevent, one level up.

THIS IS A GATE.  It exits nonzero when a crate in the population selects no
backend.  It also exits nonzero when its own self-test fails, which means the
tool is broken rather than the tree -- the two are distinguished in the output.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METAINTERP = "majit-metainterp"
BACKEND_FEATURE = re.compile(r'majit-metainterp feature "(?:dynasm|cranelift|wasm)"')


def _cargo(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    # `--locked` protects the REPO's Cargo.lock: a check must never rewrite it.
    #
    # WARNING: It is wrong for the synthesized controls, and passing it there fails
    # them in a way that reads like a finding.  A scratch crate has no lockfile
    # yet, so `--locked` refuses to resolve at all and the control reports "not
    # in the population" -- indistinguishable from the tree being clean.  The
    # flag is therefore scoped to the thing it protects.  There is no lockfile
    # in a TemporaryDirectory to protect.
    locked = ["--locked"] if cwd == REPO else []
    return subprocess.run(
        ["cargo", *args, *locked],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def in_population(crate: str, cwd: Path = REPO) -> bool:
    """Whether `crate`'s dependency graph contains majit-metainterp.

    `cargo tree -i` exits nonzero with "did not match any packages" when the
    inverted package is absent from the graph, which is the discrimination
    stage 2 cannot make for itself.
    """
    return _cargo(["tree", "-p", crate, "-i", METAINTERP], cwd).returncode == 0


def selects_backend(crate: str, cwd: Path = REPO) -> bool:
    proc = _cargo(["tree", "-p", crate, "-e", "features", "-i", METAINTERP], cwd)
    return bool(BACKEND_FEATURE.search(proc.stdout))


def _scratch_crate(root: Path, name: str, deps: str) -> Path:
    d = root / name
    (d / "src").mkdir(parents=True)
    (d / "Cargo.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "0.0.0"\nedition = "2021"\n'
        f"[workspace]\n{deps}"
    )
    (d / "src" / "main.rs").write_text("fn main() {}\n")
    return d


def self_test() -> list[str]:
    """Prove the probe discriminates, on every run.

    A sweep reporting "all clear" is worth nothing unless the probe can say
    otherwise; a positive control shows it FIRES and a negative control shows it
    stays SILENT, and neither shows it can tell two different reasons for
    silence apart.  Only case 3 does that.
    """
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # CASE 1 -- in population, NO backend.  The defect itself.
        defective = _scratch_crate(
            root,
            "backend-edge-negative-control",
            f'[dependencies]\n{METAINTERP} = {{ path = "{REPO / "majit" / METAINTERP}",'
            " default-features = false }\n",
        )
        if not in_population("backend-edge-negative-control", defective):
            failures.append(
                "case 1: the negative control is not in the population at all; "
                "the control is broken, not the tree"
            )
        elif selects_backend("backend-edge-negative-control", defective):
            failures.append(
                "case 1: a crate selecting NO backend reads as PRESENT -- the "
                "probe is blind and every result below is void"
            )

        # CASE 2 -- in population, backend present.
        if not in_population("dualtape"):
            failures.append(
                "case 2: dualtape no longer reaches " + METAINTERP + "; re-pick "
                "the positive control"
            )
        elif not selects_backend("dualtape"):
            failures.append(
                "case 2: a crate that DOES select a backend reads as absent -- "
                "the probe is inverted"
            )

        # CASE 3 -- NOT in population.  The case that broke the first version,
        # and the one no historical tree can supply.
        unrelated = _scratch_crate(root, "backend-edge-out-of-scope-control", "")
        if in_population("backend-edge-out-of-scope-control", unrelated):
            failures.append(
                "case 3: a crate with no " + METAINTERP + " edge reports as IN "
                "the population -- stage 1 does not discriminate, so stage 2's "
                "silence cannot be read"
            )
    return failures


def workspace_members() -> list[str]:
    proc = _cargo(["metadata", "--no-deps", "--format-version", "1"], REPO)
    if proc.returncode != 0:
        sys.exit(f"backend-edge: `cargo metadata` failed:\n{proc.stderr}")
    return sorted(p["name"] for p in json.loads(proc.stdout)["packages"])


def main() -> int:
    broken = self_test()
    if broken:
        print("backend-edge: SELF-TEST FAILED -- the tool is broken, not the tree")
        for f in broken:
            print(f"  {f}")
        return 2

    in_pop, out_pop, bad = [], [], []
    for crate in workspace_members():
        if not in_population(crate):
            out_pop.append(crate)
            continue
        in_pop.append(crate)
        if not selects_backend(crate):
            bad.append(crate)

    for crate in bad:
        print(
            f"FAIL {crate}: reaches {METAINTERP} and selects no backend, so "
            f"`cargo test -p {crate}` builds zero test targets. Add a backend to "
            f"its own `default`."
        )

    # THREE OUTCOMES, AND N/A IS PRINTED BY NAME.
    #
    # The denominator is printed on every run for the same reason
    # check-citation-drift.py prints its own: alone, a future `0 without a
    # backend` would be spelled identically by "the tree is clean" and "the
    # sweep did not run".
    #
    # ⭐ But a count is not enough for the N/A bucket, and this is the lesson of
    # this tool's own worst bug.  `majit-rlib` is not a pass and not a failure --
    # the predicate DOES NOT APPLY to it, because no majit-metainterp is in its
    # graph.  The first version of this check had no such outcome and would have
    # demanded a backend on a crate that must not have one.  Collapsing that
    # third state into a number puts the population beyond audit: a wrong
    # population still prints a plausible total, which is exactly how an earlier
    # hand-built census of this same set read 21 when it was 23.  ⇒ NAME THEM.
    # A reader who knows the tree can then falsify the population by reading the
    # output, instead of having to re-derive it.
    #
    # WARNING: POPULATION SCOPE: workspace members only.  Path-dependency consumers
    # outside the workspace compile against these crates and are NOT covered
    # here (task #79).  A clean run says nothing about them.
    total = len(in_pop) + len(out_pop)
    print(
        f"backend-edge: {total} workspace members -- "
        f"{len(in_pop)} in population "
        f"({len(in_pop) - len(bad)} with a backend, {len(bad)} without), "
        f"{len(out_pop)} N/A"
    )
    print(
        f"  N/A -- no {METAINTERP} in their dependency graph, so the predicate "
        f"does not apply (a backend here would be wrong, not missing):"
    )
    for crate in out_pop:
        print(f"    {crate}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
