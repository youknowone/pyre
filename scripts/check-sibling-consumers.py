#!/usr/bin/env python3
"""Compile the out-of-tree trees that path-depend on our majit crates.

WARNING: THIS IS NOT A GATE, AND MUST NOT BECOME ONE.  It cannot run in CI: the trees
it checks are separate git repos that only exist when someone has cloned them
side by side with this one.  A CI step pointed at a directory absent on the
runner either errors for the wrong reason or -- written defensively -- silently
passes, which is a brake that cannot fire.  This is a local habit, run by hand
before landing a change to a majit public API.

THE DEMONSTRATED COST
    `set_on_compile_loop` gained a fourth closure parameter.  Its 8 in-repo call
    sites were updated and mutation-proven; its 4 out-of-repo call sites were
    invisible to every gate this workspace has, and the trees stopped compiling.

WHY NOTHING SAW IT
    The consumers are separate git repos nested in the working tree, so
    `git ls-files wasmi/` is empty, `git status` never recurses, an `rg` over
    tracked files misses them, `cargo test --all` excludes them, and CI never
    checks them out.  Their manifests point three levels above their own repo
    root, so the configuration is buildable ONLY where both trees sit side by
    side -- which is why no gate in EITHER project was ever able to see it.

WHY TWO STAGES, AND WHY STAGE 1 IS THE POINT
    Stage 2 (`cargo check`) is the check.  Stage 1 classifies the dependency
    EDGE, and it is not optional, for two independent reasons.

    It is the positive control.  A green stage 2 has two causes and only one is
    good news: the tree really compiled against our live source, or the feature
    that pulls majit in was never enabled and nothing under test was built at
    all.  Stage 1 separates them before the compile, so a pass means something.

    It is also the scope rule.  Only edges that RESOLVE to our source are at
    risk from our edits; an edge that resolves to frozen git source can neither
    break nor certify, and sweeping it burns a cold build proving nothing.

STOP: AND THE MANIFEST DOES NOT DECIDE THAT -- MEASURED, IT SKIPPED THE ONE BROKEN TREE
    This classifier used to read the manifest TEXT: `git = ..., rev = ...` was
    filed PINNED and skipped.  `cel-jit`'s manifest says exactly that, and a
    gitignored `.cargo/config.toml` in that tree `[patch]`es the edge back onto
    our live source at RESOLUTION time.  So when `set_on_compile_loop` gained its
    fourth parameter, cel-jit broke -- and this script printed

        PINNED   cel-jit: rev-pinned to frozen source -- not at risk, skipped
        EXPECTED-RED wasmi-majit-pr: ... E0593 set_on_compile_loop takes 4 args

    reporting the identical break one row down while declaring cel-jit immune.
    It knew the failure; it never asked the tree that had it.

    ⇒ Stage 1 now runs in two parts: the manifest text, then `cargo metadata`
    from inside the tree, which reports each resolved package's `source` (null
    for a path, `git+...` for a pin).  RESOLUTION WINS; the text is kept only so
    a disagreement can be printed.  A disagreement is not an error -- it is the
    `[patch]` doing its job, and it is HOST-DEPENDENT: on CI there is no
    `.cargo/config.toml`, so the same tree in the same repo is genuinely PINNED
    there and LIVE here.  Both are correct, which is why every run prints which
    one it observed rather than acting on it silently.

    WARNING: The premise the text rule came from was not careless -- "`rev =` pins are
    protective" is a true statement about the RELATIONSHIP.  The tool needs the
    RESOLUTION, and a gitignored config file separates the two.  A correct model
    applied through the wrong observable.

STOP: SCOPE LIMIT -- THIS CLOSES THE COMPILE-TIME CLASS ONLY
    It is blind to runtime string contracts.  Renaming an env gate such as
    `PYRE_PORTAL_RCA`, which these trees read with their own copies of the
    literal, compiles clean on both sides and silently desynchronizes them.
    A green run here does NOT mean the exposure is covered.

EXIT CODES
    0  every LIVE tree compiled, and at least one was checked
    1  a LIVE tree failed to compile, or a known-broken tree unexpectedly passed
    2  nothing was checked -- no LIVE consumer tree was found (NOT a pass)

    WARNING: A tree whose edge could not be RESOLVED is reported as UNRESOLVED and is
    counted as neither checked nor skipped-as-safe.  It does not change the exit
    code -- the tool has no verdict to give about it -- so read the output, not
    just the status.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAJIT = REPO / "majit"

# Features a package needs beyond the one that enables majit itself.
#
# WARNING: Keyed by PACKAGE name, not directory name.  Two sibling checkouts of the
# same project both build package `wasmi`, and keying this by directory silently
# gave one of them a different feature set from the other -- an A/B where the
# arms differ in two variables at once.
#
# WARNING: Deliberately NOT `--all-features`, and deliberately no backend feature.
# wasmi's `majit-jit` already selects `majit-metainterp/cranelift` across the
# path edge, so naming a backend here would add nothing -- and supplying a
# backend explicitly is precisely what masks the defect class `check-backend-
# edge.py` exists to catch.  Keep this table minimal and justified per entry.
EXTRA_FEATURES = {
    # `wat` builds the .wat workload modules the majit-jit tests assemble
    # in-process; the majit-jit test targets declare it in required-features.
    "wasmi": ["wat"],
}

# Trees known to be broken, with the reason.  An entry here is an EXPECTED red,
# so it does not fail the run -- but if it starts PASSING that is reported as a
# failure too, because the recorded reason has gone stale and this table is
# then lying.  An oracle that cannot fail is not an oracle.
KNOWN_BROKEN = {
    "wasmi-majit-pr": (
        "TWO independent API breaks, not one -- measured, not assumed: "
        "E0593 set_on_compile_loop takes a 4-arg closure now, and E0599 "
        "JitDriver::with_options no longer exists. Repairing them needs "
        "authorization that named the wasmi/ tree only, so the asymmetry "
        "between the two checkouts is deliberate and recorded, not an oversight"
    ),
}

DEP_RE = re.compile(
    r"^\s*(?P<name>majit-[\w-]+)\s*=\s*\{(?P<body>[^}]*)\}", re.MULTILINE
)
PKG_NAME_RE = re.compile(r'^\s*name\s*=\s*"(?P<name>[^"]+)"', re.MULTILINE)
ENABLING_FEATURE_RE = re.compile(
    r"^(?P<feat>[\w-]+)\s*=\s*\[(?P<body>[^\]]*)\]", re.MULTILINE
)


def sibling_trees() -> list[Path]:
    """Top-level directories this repo cannot see into.

    `git ls-files -- <dir>` returning empty is exactly the blind spot being
    closed: a nested separate git repo is untracked here, so every repo-scoped
    census misses it.  Discovering the trees rather than hardcoding them means
    a newly cloned consumer is swept without editing this file.
    """
    out = []
    for d in sorted(REPO.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in {"target", "build"}:
            continue
        tracked = subprocess.run(
            ["git", "ls-files", "--", d.name],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if not tracked.stdout.strip():
            out.append(d)
    return out


def manifests(tree: Path) -> list[Path]:
    found = []
    for m in tree.rglob("Cargo.toml"):
        parts = set(m.parts)
        if "target" in parts or ".git" in parts:
            continue
        found.append(m)
    return found


def classify(manifest: Path) -> tuple[str, str] | None:
    """Classify a manifest's majit dependency edge.

    Returns (kind, package) where kind is 'live', 'pinned' or 'unknown', or
    None when the manifest names no majit dependency at all.
    """
    text = manifest.read_text(encoding="utf-8", errors="replace")
    kinds = set()
    for dep in DEP_RE.finditer(text):
        body = dep.group("body")
        pm = re.search(r'path\s*=\s*"([^"]+)"', body)
        if pm:
            target = (manifest.parent / pm.group(1)).resolve()
            kinds.add("live" if MAJIT in target.parents or target == MAJIT else "unknown")
        elif re.search(r"\bgit\s*=", body):
            kinds.add("pinned")
        else:
            kinds.add("unknown")
    if not kinds:
        return None
    pkg = PKG_NAME_RE.search(text)
    kind = "live" if "live" in kinds else ("pinned" if "pinned" in kinds else "unknown")
    return kind, (pkg.group("name") if pkg else manifest.parent.name)


def resolved_kind(meta: dict) -> tuple[str, str]:
    """Classify a majit edge from `cargo metadata` output, not from manifest text.

    Pure so it can be self-tested without a cargo run.  Returns (kind, evidence)
    where kind is 'live', 'pinned' or 'unknown'.

    A resolved package's `source` is null when it came from a path -- including
    a path a `[patch]` substituted for a git edge -- and `git+<url>?rev=..` when
    the pin actually held.  `unknown` when no majit package is in the graph at
    all, which is a THIRD value and not a synonym for either other one.
    """
    sources = {
        pkg.get("source")
        for pkg in meta.get("packages", [])
        if str(pkg.get("name", "")).startswith("majit-")
    }
    if not sources:
        return "unknown", "no majit-* package in the resolved graph"
    if None in sources:
        # Deliberately says nothing about *why*: a plain `path =` edge and a
        # `[patch]`ed `git =` edge are indistinguishable here and should be, the
        # point being that both resolve to our source.  The caller prints the
        # `[patch]` explanation only when the manifest text disagreed.
        return "live", "resolved to a local path"
    git = sorted(s for s in sources if s)
    return "pinned", f"resolved to {git[0]}"


def resolve_edge(tree: Path, feats: list[str]) -> tuple[str, str]:
    """Ask cargo what the tree actually resolves its majit edge to.

    WARNING: `feats` is not optional, and getting this wrong is silent.  The majit
    dependency is OPTIONAL in every consumer, so a default-feature resolve omits
    it from the graph entirely and this returns 'unknown' for a tree that is
    perfectly live -- measured on all three trees before the features were
    passed (0 majit packages with defaults, 10 with `--features majit-jit`).
    Pass the same feature list stage 2 compiles with, so both stages describe
    one configuration.

    WARNING: A failure here is reported as 'unknown', never as 'pinned'.  Falling back
    to the manifest text on error would reinstate exactly the blind spot this
    stage exists to close, and would do it silently.
    """
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--features", ",".join(feats)],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        first = next(
            (ln for ln in proc.stderr.splitlines() if ln.startswith("error")),
            proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no stderr",
        )
        return "unknown", f"cargo metadata failed: {first}"
    try:
        return resolved_kind(json.loads(proc.stdout))
    except (json.JSONDecodeError, TypeError) as exc:
        return "unknown", f"cargo metadata output unparseable: {exc}"


def enabling_feature(manifest: Path) -> str | None:
    """The feature that turns the majit dependency on.

    Derived, not hardcoded: a manifest that renames `majit-jit` keeps working,
    and a manifest where the dep became non-optional reports None rather than
    checking a feature that no longer exists.
    """
    text = manifest.read_text(encoding="utf-8", errors="replace")
    for feat in ENABLING_FEATURE_RE.finditer(text):
        if "dep:majit-metainterp" in feat.group("body"):
            return feat.group("feat")
    return None


def self_test() -> list[str]:
    """Positive control for the classifier itself.

    A stage-1 that silently classified everything as PINNED would skip every
    tree and this script would exit 2 forever, reading as "no consumers" rather
    than "the tool is broken".  Synthesize one manifest of each kind and assert
    the classifier separates them.
    """
    failures = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cases = {
            "live": f'[package]\nname = "p"\n[dependencies]\n'
            f'majit-metainterp = {{ path = "{MAJIT / "majit-metainterp"}" }}\n',
            "pinned": '[package]\nname = "p"\n[dependencies]\n'
            'majit-metainterp = { git = "https://x", rev = "abc" }\n',
        }
        for expected, body in cases.items():
            m = root / expected / "Cargo.toml"
            m.parent.mkdir(parents=True)
            m.write_text(body)
            got = classify(m)
            if got is None or got[0] != expected:
                failures.append(f"classifier: {expected} manifest read as {got}")
        empty = root / "none" / "Cargo.toml"
        empty.parent.mkdir(parents=True)
        empty.write_text('[package]\nname = "p"\n')
        if classify(empty) is not None:
            failures.append("classifier: majit-free manifest was not ignored")

    # Stage 1b's parser, over synthesized `cargo metadata` payloads.  The
    # decisive case is the third: a manifest whose TEXT says git+rev but whose
    # RESOLVED source is null, which is what a `[patch]` produces and what this
    # tool used to file as PINNED and skip.
    resolution_cases = [
        ("live", {"packages": [{"name": "majit-metainterp", "source": None}]}),
        (
            "pinned",
            {"packages": [{"name": "majit-metainterp", "source": "git+https://x?rev=abc"}]},
        ),
        (
            "live",
            {
                "packages": [
                    {"name": "majit-metainterp", "source": None},
                    {"name": "majit-ir", "source": "git+https://x?rev=abc"},
                ]
            },
        ),
        ("unknown", {"packages": [{"name": "serde", "source": "registry+https://y"}]}),
        ("unknown", {"packages": []}),
    ]
    for expected, payload in resolution_cases:
        got, _ = resolved_kind(payload)
        if got != expected:
            failures.append(f"resolution: expected {expected}, got {got} for {payload}")

    # ⭐ THE REGRESSION CASE — the exact configuration that produced the miss,
    # expressed hermetically so it does not need a `[patch]`ed tree to exist.
    # Neither stage alone can express it: the manifest text really does say
    # `rev =`, the resolution really is a path, and the DISAGREEMENT is the
    # whole finding.  Asserting only that resolution says `live` would still
    # pass if `classify` were changed to agree with it, which would delete the
    # signal that tells a reader the `[patch]` is in play at all.
    # The `rev` below is all zeros ON PURPOSE.  Nothing resolves it -- `classify`
    # only needs the manifest to SAY `rev =` -- and a real-looking sha here is a
    # false positive for every census that hunts stale commit citations, which is
    # what a plausible-looking one was until 2026-08-11.
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "Cargo.toml"
        m.write_text(
            '[package]\nname = "cel"\n[dependencies]\n'
            'majit-metainterp = { git = "https://github.com/youknowone/pyre.git",'
            ' rev = "0000000000000000000000000000000000000000" }\n'
        )
        text_kind = classify(m)
        res_kind, _ = resolved_kind({"packages": [{"name": "majit-metainterp", "source": None}]})
        if text_kind is None or text_kind[0] != "pinned":
            failures.append(f"patched-consumer: manifest text read as {text_kind}, want pinned")
        elif res_kind != "live":
            failures.append(f"patched-consumer: resolution read as {res_kind}, want live")
        elif text_kind[0] == res_kind:
            failures.append("patched-consumer: the two stages agreed -- the disagreement is the point")
    return failures


def check(tree: Path, pkg: str, feats: list[str]) -> tuple[bool, str]:
    # A target dir outside both this repo's `target/` and the sibling checkout:
    # the sibling trees are read-only to this workspace, and the shared target
    # is contended.  Set MAJIT_SIBLING_TARGET_DIR to a persistent path to keep
    # these builds incremental.
    import os

    target = os.environ.get(
        "MAJIT_SIBLING_TARGET_DIR", str(Path(tempfile.gettempdir()) / "majit-sibling")
    )
    env = {**os.environ, "CARGO_TARGET_DIR": str(Path(target) / tree.name)}
    proc = subprocess.run(
        ["cargo", "check", "-p", pkg, "--features", ",".join(feats)],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode == 0:
        return True, ""
    lines = [ln for ln in proc.stderr.splitlines() if ln.startswith("error")]
    return False, "\n".join(lines[:6]) or proc.stderr[-600:]


def main() -> int:
    failures = self_test()
    if failures:
        print("sibling-consumers: SELF-TEST FAILED -- the tool is broken, not the tree")
        for f in failures:
            print(f"  {f}")
        return 1

    print("sibling-consumers: NOT A GATE -- a local habit, not a CI check.")
    print(f"  repo: {REPO}\n")

    checked = 0
    bad = 0
    unresolved = 0
    for tree in sibling_trees():
        edges = [(m, c) for m, c in ((m, classify(m)) for m in manifests(tree)) if c]
        text_live = [(m, c) for m, c in edges if c[0] == "live"]
        text_pinned = [(m, c) for m, c in edges if c[0] == "pinned"]
        if not text_live and not text_pinned:
            continue

        # The manifest, the package and the feature come first: stage 1b needs
        # the feature to resolve at all, because the majit edge is optional.
        manifest, (_, pkg) = (text_live or text_pinned)[0]
        feat = enabling_feature(manifest)
        if feat is None:
            print(f"  WARNING: SKIP   {tree.name}: no feature gates dep:majit-metainterp")
            print("           the dep may have become unconditional -- check by hand")
            continue
        feats = [feat, *EXTRA_FEATURES.get(pkg, [])]

        # Stage 1b.  The manifest text is now only a second opinion: it is
        # printed when it disagrees, and it never decides.
        text_kind = "live" if text_live else "pinned"
        res_kind, evidence = resolve_edge(tree, feats)
        if res_kind != text_kind:
            print(
                f"  ⓘ RESOLVED {tree.name}: manifest text reads {text_kind.upper()}, "
                f"cargo resolves {res_kind.upper()} -- {evidence}"
            )
            print("           resolution wins. This differs per host: a gitignored")
            print("           .cargo/config.toml [patch] is absent on CI, so the same")
            print("           tree is legitimately PINNED there and LIVE here.")

        if res_kind == "unknown":
            unresolved += 1
            print(f"  WARNING: UNKNOWN {tree.name}: cannot resolve its majit edge -- {evidence}")
            print("           neither checked nor skipped-as-safe; resolve it by hand.")
            continue
        if res_kind == "pinned":
            print(f"  PINNED   {tree.name}: {evidence} -- not at risk, skipped")
            continue

        expected_broken = tree.name in KNOWN_BROKEN
        print(f"  checking {tree.name}: cargo check -p {pkg} --features {','.join(feats)}")
        ok, err = check(tree, pkg, feats)
        checked += 1

        if ok and not expected_broken:
            print(f"  OK       {tree.name}")
        elif ok and expected_broken:
            bad += 1
            print(f"  WARNING: STALE  {tree.name}: recorded as known-broken but it COMPILES.")
            print(f"           recorded reason: {KNOWN_BROKEN[tree.name]}")
            print("           remove it from KNOWN_BROKEN -- this table is now lying.")
        elif expected_broken:
            print(f"  EXPECTED-RED {tree.name}: {KNOWN_BROKEN[tree.name]}")
            for ln in err.splitlines():
                print(f"           {ln}")
        else:
            bad += 1
            print(f"  STOP: BROKEN {tree.name}")
            for ln in err.splitlines():
                print(f"           {ln}")

    print()
    if unresolved:
        print(f"sibling-consumers: WARNING: {unresolved} tree(s) UNRESOLVED -- their coverage is")
        print("  unknown, which is neither a pass nor a skip. Read the lines above.")
    if checked == 0:
        print("sibling-consumers: NOTHING WAS CHECKED -- no live consumer tree found.")
        print("  This is not a pass. Clone the consumer repos beside this one to run it.")
        return 2
    if bad:
        return 1
    print(f"sibling-consumers: {checked} tree(s) checked, all as recorded.")
    print("  STOP: Compile-time only. Renaming an env gate string still desyncs silently.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
