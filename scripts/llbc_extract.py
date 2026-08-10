"""Charon ULLBC extraction engine with source-fingerprint skip logic.

Import-only library: a per-repo driver declares its crate table as a dict of
`CrateSpec`s and calls `run_cli(...)`. This module carries ZERO crate names —
every crate (its dir, cargo flags, fingerprint inputs, output artefact) is
declared by the driver, so the engine stays neutral about which consumer
(pyre, or an external interpreter crate) it is extracting for.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# Domain separator for the bytes hashed by `source_fingerprint`. Bump this
# only when the fingerprint algorithm or the meaning of its input set changes.
# Ordinary refactors, diagnostics and comments in this file do not change what
# Charon compiles and must not invalidate every multi-minute LLBC artefact.
FINGERPRINT_SCHEMA = "2"


@dataclass
class CrateSpec:
    """One extractable crate.

    - `crate_dir`: absolute dir where `charon cargo` runs (holds Cargo.toml/src).
    - `output_name`: artefact filename under the driver's output dir
      (e.g. `<crate>.ullbc`).
    - `cargo_args`: extra flags passed after `--`; each arg may contain the
      `{features}` placeholder, substituted with the active cargo feature set
      (e.g. `["--features", "{features}"]` or `["--no-default-features"]`).
    - `charon_args`: extra flags passed to Charon itself, before the `--`
      separator (e.g. `["--include", "somecrate::module::_"]` to translate the
      bodies of items in a foreign dependency instead of keeping them opaque).
      Same `{features}` placeholder substitution as `cargo_args`.
    - `fingerprint_pathspecs`: explicit git pathspecs (relative to the driver's
      `root`) that fingerprint this crate's sources. `None` derives them from a
      `cargo metadata` dependency walk instead.
    - `excluded_deps`: path-dependency package names dropped from this crate's
      fingerprint because the artefact holds zero references to them; the
      extraction guard re-checks the artefact and fails loud if that drifts.
      Naming a package drops its EXCLUSIVELY-REACHED SUBTREE too — anything
      reachable only by going through it. A package some non-excluded parent
      also reaches is kept, so listing a widely-shared package removes only
      its own files. The guard checks the NAMED packages; the subtree is
      covered by inheritance (see `_collect_inputs`), which is why the two
      are not, and must not be, the same set.
    - `layout_targets`: target triples, besides the extraction host, this
      crate also emits a layout sidecar for (see `layout_sidecar_name`).
      `None` takes the driver's default; `()` opts out. Every listed target
      must compile under `layout_cargo_args`.
    - `layout_cargo_args`: `cargo_args` replacement for the sidecar passes.
      A cross target usually needs a different feature set than the host
      (a native code generator does not build for wasm32). `None` reuses
      `cargo_args`.
    """

    name: str
    crate_dir: Path
    output_name: str
    cargo_args: list[str] = field(default_factory=list)
    charon_args: list[str] = field(default_factory=list)
    fingerprint_pathspecs: list[str] | None = None
    excluded_deps: set[str] = field(default_factory=set)
    layout_targets: tuple[str, ...] | None = None
    layout_cargo_args: list[str] | None = None

    def layout_sidecar_name(self, target: str) -> str:
        """Artefact name of this crate's `target` layout sidecar."""
        return f"{Path(self.output_name).stem}.{target}.layouts.ullbc"


@dataclass
class Engine:
    """Resolved driver configuration threaded through the extraction helpers."""

    specs: dict[str, CrateSpec]
    default_crates: list[str]
    root: Path
    out_dir: Path
    base_pathspecs: list[str]
    charon_root: Path
    cargo_features: str
    extraction_abi: str
    metadata_feature_crates: tuple[str, ...] = ()
    layout_targets: tuple[str, ...] = ()
    layout_target_rustflags: str = ""
    # Files/directories a driver declares by hand because the git channel
    # cannot carry them. Two kinds, and BOTH are hashed by content:
    #
    #   * outside `root` — the engine module itself, a sibling checkout. `git
    #     ls-files` refuses a pathspec that leaves the repository.
    #   * inside `root` but IGNORED — an uncommitted `.cargo/config.toml`, a
    #     lockfile a repo gitignores. `_collect_inputs` resolves pathspecs
    #     through `ls-files ∪ ls-files --others --exclude-standard`, and an
    #     ignored file is in neither, so declaring it as a pathspec is inert.
    #     `refuse_inert_pathspecs` names this field as the remedy, so keep it
    #     accepting in-root paths: labels come from `os.path.relpath`, which
    #     spells an in-root path without any `..`.
    #
    # Patched path deps need no declaration — `_collect_inputs` discovers those.
    external_inputs: tuple[Path, ...] = ()

    def spec(self, crate: str) -> CrateSpec:
        try:
            return self.specs[crate]
        except KeyError:
            known = " ".join(self.specs)
            raise SystemExit(
                f"extract-llbc.py: unknown crate '{crate}'\n  known: {known}"
            )


def excluded_packages(eng: Engine, crates: list[str]) -> set[str]:
    """Packages to drop from the combined fingerprint of `crates`.

    A package is dropped only when EVERY requested crate excludes it, so a
    multi-crate fingerprint (e.g. a combined `--fingerprint a b c` call) stays
    conservative whenever any crate in the set still depends on it.
    """
    sets = [eng.spec(crate).excluded_deps for crate in crates]
    return set.intersection(*sets) if sets else set()


def platform_info() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "darwin-arm64", "charon"
    if system == "Darwin" and machine == "x86_64":
        return "darwin-x86_64", "charon"
    if system == "Linux" and machine in {"arm64", "aarch64"}:
        return "linux-aarch64", "charon"
    if system == "Linux" and machine == "x86_64":
        return "linux-x86_64", "charon"
    if system == "Windows" or system.startswith(("MSYS", "MINGW", "CYGWIN")):
        return "windows", "charon.exe"
    raise SystemExit(f"extract-llbc.py: unsupported platform {system}-{machine}")


def expand_features(arg: str, cargo_features: str) -> str:
    features = [f.strip() for f in cargo_features.split(",") if f.strip()]
    if "{features}" not in arg or len(features) <= 1:
        # No placeholder, or a single/absent feature: whole-string
        # substitution already yields the right flag.
        return arg.format(features=cargo_features)
    # Multiple features: a template like `crate/{features}` prefixes the
    # placeholder, so splicing the raw `a,b` list into one slot only
    # prefixes the first feature. Expand the template per feature and
    # rejoin so each feature keeps the prefix.
    return ",".join(arg.format(features=feature) for feature in features)


def crate_flags(spec: CrateSpec, cargo_features: str) -> list[str]:
    return [expand_features(arg, cargo_features) for arg in spec.cargo_args]


def charon_crate_flags(spec: CrateSpec, cargo_features: str) -> list[str]:
    return [expand_features(arg, cargo_features) for arg in spec.charon_args]


def crate_layout_flags(
    spec: CrateSpec, cargo_features: str, flags: list[str]
) -> list[str]:
    """Cargo args for this crate's layout sidecar passes.

    `layout_cargo_args` when the spec declares them (a cross target usually
    needs a different feature set), otherwise the host extraction `flags`.
    """
    if spec.layout_cargo_args is None:
        return flags
    return [expand_features(arg, cargo_features) for arg in spec.layout_cargo_args]


def features_in_cargo_flags(crate: str, flags: list[str]) -> list[str]:
    """Feature names a cargo arg list enables, workspace-qualified.

    A bare name is crate-relative (the extraction passes run cargo inside the
    crate directory) and must be spelled `crate/feature` for the
    workspace-level `cargo metadata`. `--no-default-features` is ignored:
    dropping it can only widen the graph, and the fingerprint may
    over-approximate but must never miss an input.
    """
    features: list[str] = []
    index = 0
    while index < len(flags):
        arg = flags[index]
        if arg in ("--features", "-F") and index + 1 < len(flags):
            value = flags[index + 1]
            index += 2
        elif arg.startswith(("--features=", "-F=")):
            value = arg.split("=", 1)[1]
            index += 1
        else:
            index += 1
            continue
        for feature in value.replace(",", " ").split():
            features.append(feature if "/" in feature else f"{crate}/{feature}")
    return features


def run_capture(args: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ).stdout


_host_triple_cache: dict[Path, str] = {}


def host_triple(root: Path) -> str:
    if value := os.environ.get("LLBC_TARGET_TRIPLE"):
        return value
    cache_key = root.resolve()
    if value := _host_triple_cache.get(cache_key):
        return value
    rustc_info = run_capture(["rustc", "-vV"], cwd=root)
    for line in rustc_info.splitlines():
        if line.startswith("host: "):
            value = line.removeprefix("host: ")
            _host_triple_cache[cache_key] = value
            return value
    raise SystemExit("extract-llbc.py: could not determine rustc host triple")


_metadata_cache: dict[tuple, dict] = {}


def metadata(
    eng: Engine,
    cargo_features: str,
    platform: str,
    extra_features: tuple[str, ...] = (),
) -> dict:
    """`cargo metadata` filtered to `platform`, memoised per full query.

    The platform matters: `--filter-platform` drops dependencies gated on other
    targets, so the host's graph does not see a `[target.'cfg(...)']` path dep
    that only the cross-target layout sidecar pass compiles.
    `extra_features` carries workspace-qualified features a layout sidecar
    pass enables beyond the host pass (`layout_cargo_args`), so the graph
    sees the dependencies they pull in. The memo key spans every input that
    shapes the result — including the engine's workspace root and feature
    crates, which differ between engines sharing this module.
    """
    key = (
        str(eng.root),
        eng.metadata_feature_crates,
        cargo_features,
        platform,
        extra_features,
    )
    if key in _metadata_cache:
        return _metadata_cache[key]

    metadata_features: list[str] = []
    for feature in cargo_features.split(","):
        feature = feature.strip()
        if feature:
            metadata_features.extend(
                f"{crate}/{feature}" for crate in eng.metadata_feature_crates
            )
    metadata_features.extend(extra_features)

    args = [
        "cargo",
        "metadata",
        "--format-version=1",
        "--filter-platform",
        platform,
    ]
    if metadata_features:
        args.extend(["--features", ",".join(metadata_features)])
    result = json.loads(run_capture(args, cwd=eng.root))
    _metadata_cache[key] = result
    return result


_INCLUDE_CALL = re.compile(r"\binclude(?:_str|_bytes)?!\s*\(\s*")
# A plain, escape-free string literal is the only spelling this resolves. A raw
# literal or one carrying a `\` escape falls through to the refusal below rather
# than being decoded wrongly.
_INCLUDE_PATH = re.compile(r"\"([^\"\\]*)\"")


def include_closure(root: Path, files: set[Path]) -> set[Path]:
    """The files `include*!` reads that no crate pathspec covers.

    `include!` / `include_str!` / `include_bytes!` resolve their argument
    against the directory of the file holding the macro, so an argument that
    climbs out of the crate is still a compiler input while sitting outside
    every pathspec the enumeration builds. `majit-metainterp`'s `REAL_RULES`
    is the live instance: it includes PyPy's `real.rules` out of the vendored
    `rpython/` tree, and editing that file changes what the crate compiles
    without moving a digest that never hashed it.

    Only a plain string literal is resolvable from the text, so an argument in
    any other spelling stops extraction instead of being skipped: a skip is
    indistinguishable from a fresh artefact, which is the failure this walk
    exists to remove. The two spellings in the tree today are allowed by shape
    and each is checked below; a third has to be audited the same way before it
    can pass.

    An argument matched inside a comment or a string names either nothing or a
    file the crate does not read; both only widen the digest, which costs a
    re-extraction and never a wrong offset.

    `#[path]` is the other way a crate compiles a file outside its own tree and
    is deliberately not walked: of the four in the workspace, the only pair
    whose `mod` file any crate enumerates redirects into `pyre-jit-trace/src/`,
    which that same crate's closure already covers. Walk it when a redirection
    appears that a pathspec does not reach.

    A refusal raised here reaches whoever runs extraction and whoever runs
    `--fingerprint` from CI, but not `pyre-jit-trace`'s build script: that one
    collapses every non-answer from this driver to silence on purpose
    (`pyre/pyre-jit-trace/build.rs` `llbc_fingerprint_output`).
    """
    extra: set[Path] = set()
    pending = [path for path in files if path.suffix == ".rs"]
    queued = set(pending)
    while pending:
        rel_source = pending.pop()
        source = root / rel_source
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        for call in _INCLUDE_CALL.finditer(text):
            line = text.count("\n", 0, call.start()) + 1
            where = f"{rel_source.as_posix()}:{line}"
            literal = _INCLUDE_PATH.match(text, call.end())
            if literal is None:
                _reject_unresolvable_include(where, text[call.end() : call.end() + 80])
                continue
            target = (source.parent / literal.group(1)).resolve()
            if not target.is_file():
                continue
            try:
                rel = target.relative_to(root)
            except ValueError:
                raise SystemExit(
                    f"extract-llbc.py: {where} includes {target}, which is "
                    "outside the repository. The digest is keyed by "
                    "repo-relative paths, so that file cannot be hashed and "
                    "the artefact would read as fresh across edits to it."
                ) from None
            if rel in files or rel in extra:
                continue
            extra.add(rel)
            if rel.suffix == ".rs" and rel not in queued:
                queued.add(rel)
                pending.append(rel)
    return extra


def _reject_unresolvable_include(where: str, tail: str) -> None:
    """Let through the spellings whose inputs are covered; refuse the rest."""
    # `include!()` with no argument only occurs in prose about the macro.
    if tail.startswith(")"):
        return
    # `concat!(env!("OUT_DIR"), ...)` names a file the build script generates
    # into the cargo out dir, from sources this enumeration already carries.
    if tail.startswith("concat!") and 'env!("OUT_DIR")' in tail:
        return
    # A `macro_rules!` parameter, resolved at each call site. The one instance
    # is the `appleveldefs:` arm of `pyre_module_init!`, whose call sites pass
    # sibling `app_*.py` names that the crate's own `src/` pathspec enumerates.
    if tail.startswith("$"):
        return
    raise SystemExit(
        f"extract-llbc.py: {where} includes `{tail.splitlines()[0].strip()}`, "
        "which this fingerprint cannot resolve to a path. Whatever it names "
        "would be a compiler input the digest never hashes, so the artefact "
        "would read as fresh across edits to it. Either give it a plain "
        "string literal or add its shape to _reject_unresolvable_include with "
        "the reason its inputs are already enumerated."
    )


def refuse_inert_pathspecs(eng: Engine) -> None:
    """Refuse a declared fingerprint input that cannot contribute anything.

    A pathspec is an input only if git will list a file under it, because
    `_collect_inputs` resolves the in-root channel through `git ls-files` and
    nothing else. A pathspec matching zero files is therefore a silent no-op —
    and a no-op that READS as coverage everywhere a human looks: in the
    driver's list, in the comments that cite it, and in the design decisions
    that rest on it. A driver carried `Cargo.lock` in exactly that state while
    a comment eight lines above called it a declared input and used it to
    justify skipping the more expensive `cargo metadata` walk, so the cheaper
    option was resting on coverage that did not exist.

    This lives in the ENGINE rather than in one driver because every consumer
    of the engine has the same hole: the check was first written against one
    driver's `BASE_PATHSPECS`, and each other driver kept the defect until it
    grew its own copy. Refusing here catches the whole class at the one place
    that resolves pathspecs, and it would have fired the day either known
    offender was added.

    Scope is the DECLARED pathspecs — `base_pathspecs`, and each spec's
    `fingerprint_pathspecs`. The ones `_collect_inputs` discovers from `cargo
    metadata` are facts about the build rather than a human's claim of
    coverage, and refusing on those would fail a driver over a dependency's
    layout it does not control. Every spec is checked, not only the crates
    requested on this run: a pathspec that rots in a spec nobody asked for
    today is then found on the next run, rather than the next time that one
    crate happens to be extracted.

    ⛔ The DECISION is git's, and only git's. `Path.exists()` below feeds the
    MESSAGE and nothing else — do not "simplify" this to an existence test. An
    ignored file EXISTS on disk and is invisible to `git ls-files`, which is
    the entire defect. Filesystem-reachable and git-reachable are different
    predicates, and a check that conflates them answers a question nobody
    asked. (The defect is not `exists()`; it is using `exists()` to answer a
    question about git. An `is_file()` test on a module about to be imported is
    a genuine filesystem question and stays correct.)

    ⭐ The contrasting sibling now lives in ANOTHER REPO, so neither file shows
    both halves of the rule on its own. A driver that declares
    `external_inputs` may legitimately use `exists()` as a VERDICT on those,
    and cel-jit's `refuse_absent_external_inputs` does: they are hashed by
    reading their bytes, never through git, so the filesystem is the correct
    oracle and the only one. Read alone, that function looks like the mistake
    this one refuses; read alone, this one looks like a blanket ban on
    `exists()`. Neither reading is right — the oracle has to match the channel
    the input is carried on, and the two guards are on different channels.

    `ls-files ∪ ls-files --others --exclude-standard` is the definition of
    git-reachable used here, matching `_collect_inputs` exactly. `ls-files
    --error-unmatch` alone is TRACKED-only, and would call a brand-new unadded
    file inert when the engine does hash it.

    The two failure modes need different remedies, so they are reported apart:
    a path that EXISTS but is ignored is a coverage hole, a path that does not
    exist is a typo or a file that moved.
    """

    def git_lists(*args: str) -> bool:
        result = subprocess.run(
            ["git", "-C", str(eng.root), *args],
            check=True,
            stdout=subprocess.PIPE,
        )
        return bool(result.stdout.strip())

    declared: list[tuple[str, list[str]]] = [("base_pathspecs", eng.base_pathspecs)]
    for name, spec in eng.specs.items():
        if spec.fingerprint_pathspecs is not None:
            declared.append(
                (f"specs[{name!r}].fingerprint_pathspecs", spec.fingerprint_pathspecs)
            )

    inert: list[tuple[str, str, bool]] = []
    for label, pathspecs in declared:
        for pathspec in pathspecs:
            # `--others` only when the tracked query came back empty: the live
            # case is one subprocess per pathspec instead of two, and this runs
            # on every invocation including the `--check` a build script makes.
            if git_lists("ls-files", "--", pathspec):
                continue
            if git_lists("ls-files", "--others", "--exclude-standard", "--", pathspec):
                continue
            inert.append((label, pathspec, (eng.root / pathspec).exists()))
    if not inert:
        return

    lines = ["extract-llbc.py: declared fingerprint pathspecs that match no file:"]
    for label, pathspec, on_disk in inert:
        if on_disk:
            lines.append(
                f"  {label}: {pathspec!r} — EXISTS on disk but git will not list"
                f" it (ignored?), so it contributes nothing to the fingerprint."
                f" Either stop declaring it or cover it another way — an"
                f" out-of-root or ignored file goes in `external_inputs=`,"
                f" which is hashed by content; do not leave it here reading as"
                f" coverage."
            )
        else:
            lines.append(
                f"  {label}: {pathspec!r} — does not exist. Renamed, moved, or a"
                f" typo: whatever it was meant to fingerprint is now"
                f" unfingerprinted."
            )
    raise SystemExit("\n".join(lines))


def _package_closure(
    target_ids: list[str],
    by_id: dict,
    resolve_nodes: dict,
    exclude: set[str],
) -> list[dict]:
    """Path-dependency packages reachable from `target_ids` avoiding `exclude`.

    Pure graph reachability over cargo-metadata shapes: `by_id` maps package id
    to package, `resolve_nodes` maps package id to its resolve node. Kept
    separate from `_collect_inputs` so `--self-test` can drive it on a synthetic
    graph — the property below is about the WALK, and on a real tree it is
    invisible, because a fingerprint that silently includes four extra files
    still hashes to something and still compares equal to itself.

    ⭐ The exclusion is applied HERE, in the walk, and not in the emission loop
    in `_collect_inputs` — excluding a package drops its whole EXCLUSIVELY-
    REACHED SUBTREE, not just its own files.

    This used to filter only at emission, which dropped the named package's
    sources and then walked through it anyway, so every dependency reachable
    ONLY through it stayed in the fingerprint. Measured on this tree:
    `pyre-interpreter` excludes `majit-translate`, and `majit-charon-reader` —
    whose sole dependent-of-record IS `majit-translate` — rode in behind it, 4
    files that no artefact references and that moved the fingerprint 3 times in
    30 days.

    `continue` before appending is what makes it a subtree drop: the node is
    neither emitted nor traversed. A package a NON-excluded parent also reaches
    is still pushed from that parent, so this computes "reachable by a path
    avoiding every excluded package" — which is the property the safety
    argument below needs, and it holds regardless of pop order. Both halves are
    load-bearing and `--self-test`'s diamond case checks them together: drop
    what only the excluded package reaches, keep what something else does.

    ⛔ Do NOT ALSO widen `extract`'s artefact guard to the packages dropped
    here. That guard's evidence is a substring search for the package's
    underscored name, and its power is PER SYMBOL: `majit_translate` occurs 373
    times in `pyre-jit.ullbc` (which excludes nothing), so the guard is
    demonstrably able to fail for it. `majit_charon_reader` occurs 0 times in
    ALL SIX artefacts this tree builds, `pyre-jit.ullbc` included — there is no
    artefact that can serve as its positive control, so a widened guard would
    pass for a reason unrelated to safety and read as verification.

    The transitive drop is certified by INHERITANCE instead, and that is a
    stronger argument than the substring test: the parent's guard establishes
    the artefact holds zero references to the excluded package, and a package
    reachable only through it cannot appear without it. The guard runs on every
    extraction, so the certificate stays live — if the artefact ever does
    reference the parent, the parent's guard fires and the exclusion (subtree
    included) has to go.
    """
    seen: set[str] = set()
    stack = list(target_ids)
    closure = []
    while stack:
        package_id = stack.pop()
        if package_id in seen:
            continue
        seen.add(package_id)
        package = by_id[package_id]
        if package["name"] in exclude:
            continue
        closure.append(package)

        for dep in resolve_nodes.get(package_id, {}).get("deps", []):
            dep_kinds = dep.get("dep_kinds", [])
            # An empty `dep_kinds` is a normal (non-dev) edge; only drop deps
            # whose every listed kind is `dev`.
            if dep_kinds and all(kind.get("kind") == "dev" for kind in dep_kinds):
                continue
            dep_package = by_id.get(dep["pkg"])
            if dep_package is not None and dep_package.get("source") is None:
                stack.append(dep_package["id"])
    return closure


def _collect_inputs(
    eng: Engine, crates: list[str], cargo_features: str
) -> tuple[list[str], list[Path]]:
    """Split the fingerprint's inputs by which channel can carry them.

    Returns `(pathspecs, external)`:

    * `pathspecs` — repo-relative, resolved through `git ls-files` below, so
      the working tree (including untracked-not-ignored files) is what gets
      hashed.
    * `external` — absolute paths the git channel cannot carry: anything
      outside `eng.root`, which `git ls-files` refuses to name at all, plus
      whatever a driver declares in `external_inputs` (which is also how an
      in-root but IGNORED file gets covered). Hashed by content instead, by
      `external_fingerprint`.

    The second list used to be dropped on the floor. The dependency walk below
    admits path deps by `source is None`, which is how cargo spells BOTH an
    in-tree workspace member and a `[patch]` redirect into a sibling checkout —
    so a patched dependency was discovered, found to be out-of-root, and
    discarded with no diagnostic. cel-jit patches `majit-{ir,macros,metainterp}`
    to `../majit/*`, which pulls the rest of that workspace in by path, and
    most of what that pulls landed here, `majit-macros` among them.

    ⛔ NO NUMBER IS QUOTED FOR THAT CLOSURE, deliberately. Two careful
    measurements of it disagree (11 members / 1 kept, and 10 packages / 9
    out-of-root) and the disagreement is UNEXPLAINED — a `CARGO_FEATURES`
    difference was proposed and then refuted, because swapping one backend
    crate for another changes membership without changing the count. The
    closure is not a stable set, and a figure written down here would be
    quoted back as if it were. This is the empirical case for a driver
    declaring a whole directory in `external_inputs` rather than a per-crate
    list the walk derived.

    `majit-macros` is a proc macro, so its expansion IS the extracted crate's
    item bodies. It was dropped by TWO independent gates, and finding the
    first hid the second: this out-of-root discard, and the target-kind filter
    below, which admitted neither `proc-macro` nor `cdylib` until it was
    inverted into a deny-list. Both are fixed; do not read that as the class
    being closed, since each was found only after the previous one was.
    """
    root = eng.root
    target_names: list[str] = []
    pathspecs = list(eng.base_pathspecs)
    external: list[Path] = [Path(p).resolve() for p in eng.external_inputs]

    for crate in crates:
        spec = eng.spec(crate)
        if spec.fingerprint_pathspecs is not None:
            pathspecs.extend(spec.fingerprint_pathspecs)
        else:
            target_names.append(crate)

    if target_names:
        # The stamp covers the host artefact AND every layout sidecar, so the
        # fingerprint must hash the union of the closures those passes compile:
        # a path dep gated on a cross target alone would otherwise change
        # without invalidating the sidecar it shaped.
        platforms = {host_triple(root)}
        for name in target_names:
            platforms.update(crate_layout_targets(eng, eng.spec(name)))

        # A layout sidecar pass may enable features the host pass does not
        # (`layout_cargo_args`); fold them into every metadata query so the
        # union closure sees the dependencies those features pull in.
        layout_features: set[str] = set()
        for name in target_names:
            spec = eng.spec(name)
            if spec.layout_cargo_args is not None:
                layout_features.update(
                    features_in_cargo_flags(
                        name, crate_layout_flags(spec, cargo_features, [])
                    )
                )
        extra_features = tuple(sorted(layout_features))

        # Never drop a requested target crate, only its excluded dependencies.
        exclude = excluded_packages(eng, crates) - set(target_names)

        for platform in sorted(platforms):
            meta = metadata(eng, cargo_features, platform, extra_features)
            packages = meta["packages"]
            by_name = {package["name"]: package for package in packages}
            by_id = {package["id"]: package for package in packages}
            resolve_nodes = {
                node["id"]: node for node in meta.get("resolve", {}).get("nodes", [])
            }

            missing = [name for name in target_names if name not in by_name]
            if missing:
                raise SystemExit(
                    "extract-llbc.py: unknown crate(s): " + ", ".join(sorted(missing))
                )

            closure = _package_closure(
                [by_name[name]["id"] for name in target_names],
                by_id,
                resolve_nodes,
                exclude,
            )

            for package in closure:
                package_dir = Path(package["manifest_path"]).resolve().parent
                if package_dir.is_relative_to(root):
                    rel_dir = package_dir.relative_to(root).as_posix()
                    pathspecs.append(f"{rel_dir}/Cargo.toml")
                else:
                    external.append(package_dir / "Cargo.toml")
                for target in package["targets"]:
                    kinds = set(target["kind"])
                    # ⭐ A DENY-list, and the direction is the whole point. This
                    # was `{"lib","bin","custom-build"} & kinds` — an allow-list
                    # written from a remembered vocabulary rather than a
                    # measured one, and it silently dropped THREE path-dep
                    # packages in this tree: `majit-macros` and `pyre-macros`
                    # (`proc-macro`) and `pyre-wasm` (`cdylib`). A proc macro is
                    # the WORST possible omission — its expansion is inlined
                    # into the consuming crate, so its sources are more coupled
                    # to the artefact's item bodies than an ordinary `lib`'s
                    # are. An allow-list that omits exactly that is the inverse
                    # of the risk ordering.
                    #
                    # The package loop above appends `Cargo.toml` BEFORE this
                    # loop runs, which is what made the omission invisible: the
                    # dropped crate still showed up in `--list-inputs`, so it
                    # read as covered while none of its sources were hashed.
                    #
                    # Naming what is NOT compiled into the artefact is a short,
                    # closed list; naming what IS grows silently with every
                    # crate type cargo adds. Getting this wrong now costs a
                    # re-extraction, never a wrong answer — the same direction
                    # the untracked-file leg below is deliberately biased in.
                    if kinds & {"example", "test", "bench"}:
                        continue
                    src_path = Path(target["src_path"]).resolve()
                    if src_path.is_relative_to(root):
                        rel_src = src_path.relative_to(root).as_posix()
                        pathspecs.append(rel_src)
                        if "custom-build" not in kinds:
                            pathspecs.append(str(Path(rel_src).parent) + "/")
                    else:
                        # Same shape as the in-root arm, one channel over: the
                        # entry point always, its directory only for a real
                        # target. A `custom-build`'s src_path is the crate's
                        # own `build.rs`, so its parent is the CRATE ROOT —
                        # walking that would pull in `target/` and every
                        # sibling target's sources.
                        external.append(src_path)
                        if "custom-build" not in kinds:
                            external.append(src_path.parent)

    # git is the enumerator, but the index is not the input set: Charon and
    # rustc read the working tree, so a source file that is on disk and not yet
    # added belongs in the digest. `--cached` alone leaves it out, and a crate
    # extracted while one of its files was untracked then fingerprints as fresh
    # against sources the artefact was never built from — staging that same
    # file later flips the identical bytes to STALE.
    #
    # `--exclude-standard` is what keeps the generated trees out. Everything it
    # drops under these pathspecs today is produced by a build —
    # `majit/charon-corpus/{target/,Cargo.lock}` and `pyre/check.snap/` — and
    # nothing a crate compiles.
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *pathspecs,
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    files = {
        Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw
    }
    files |= include_closure(root, files)
    return sorted(files, key=lambda path: path.as_posix()), external


def fingerprint_inputs(eng: Engine, crates: list[str], cargo_features: str) -> list[Path]:
    """Repo-relative inputs hashed into `source=`."""
    return _collect_inputs(eng, crates, cargo_features)[0]


def external_input_groups(
    eng: Engine, crates: list[str], cargo_features: str
) -> list[tuple[str, list[tuple[str, Path]]]]:
    """Out-of-root inputs GROUPED by the root that declared or discovered them.

    `[(root label, [(file label, absolute path), …]), …]`, sorted by root label
    and, within a root, by file label.

    Deliberately NOT named `external_inputs`: that is the driver-facing keyword
    of `run_cli`, and a parameter of that name shadows the module-level
    function for the whole body.

    ⭐ The GROUP, not the file and not the whole set, is the unit `external=`
    records a digest for, and that choice is the only reason `check` can name a
    mover. One digest over every out-of-root file fires identically whether a
    doc comment moved in a crate the artefact barely references or
    `majit-macros` did — a proc macro whose expansion IS the extracted crate's
    item bodies, so its movement voids every measurement already read out of
    the artefact. Same remedy (`--force`), very different consequence, and a
    single digest cannot tell them apart. Per FILE is the other extreme and no
    better: cel's declared closure is hundreds of files, and a report naming
    hundreds of movers carries the same zero bits spelled longer.

    Roots are deduped by label — the metadata walk runs once per layout
    platform and rediscovers the same packages — but files are NOT deduped
    across roots. `<crate>/src/lib.rs` and `<crate>/src/` are both roots the
    walk emits, so an edit to `lib.rs` moves both, and that pair of movers is
    the narrowing a reader wants rather than noise.

    The label — not the absolute path — is what enters the digest, so two
    checkouts of the same layout in different directories agree. Labels are
    relative to the artefact root, so a stamp survives relocating the whole
    cohabitation (`<super>/{pyre,cel-jit,majit}`) as a unit; it does not
    survive rearranging those repos RELATIVE to each other, which is the
    correct sensitivity, because that changes which sources are compiled.

    Directories are expanded here rather than at collection time so the walk
    sees the working tree at hashing time, matching `source=`'s
    save-not-commit behaviour for the in-root half.
    """
    _, roots = _collect_inputs(eng, crates, cargo_features)

    def label_for(path: Path) -> str:
        return Path(os.path.relpath(path, eng.root)).as_posix()

    groups: dict[str, list[tuple[str, Path]]] = {}
    for entry in roots:
        root_label = label_for(entry)
        if root_label in groups:
            continue
        if entry.is_dir():
            found: dict[str, Path] = {}
            for sub in entry.rglob("*"):
                # `target/` is build output and dot-dirs are VCS/tooling state;
                # neither is a compiler input, and `target/` alone is tens of GB.
                parts = set(sub.relative_to(entry).parts[:-1])
                if "target" in parts or any(p.startswith(".") for p in parts):
                    continue
                if sub.is_file():
                    found[label_for(sub)] = sub
            groups[root_label] = sorted(found.items())
        else:
            # A file, or a declared root that does not exist. The absent case
            # still gets a group, so it moves the digest when it appears —
            # exactly as a deleted tracked file does in `source=`.
            groups[root_label] = [(root_label, entry)]
    return sorted(groups.items())


def external_group_digest(root_label: str, files: list[tuple[str, Path]]) -> str:
    """Digest of one out-of-root group.

    The root label is folded in first so an EMPTY group (a declared directory
    holding no file the filter admits) still gets a digest of its own rather
    than colliding with every other empty group.
    """
    digest = hashlib.sha256()
    digest.update(root_label.encode("utf-8"))
    digest.update(b"\0")
    for label, path in files:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<absent>")
        digest.update(b"\0")
    return digest.hexdigest()


def external_fingerprint(eng: Engine, crates: list[str], cargo_features: str) -> str:
    """`external=`'s stamp value: one `<root label>=<digest>` per group.

    Empty is the common case (a consumer whose whole dependency closure lives
    inside its own repo, which is every pyre crate today) and it is spelled as
    an empty value rather than a digest-of-nothing, so the stamp says plainly
    that nothing outside the repo was folded in.

    `shlex.quote` per entry, space-separated: a root label is a filesystem path
    and may hold a space or a quote, and the value has to survive as ONE line
    of a line-oriented stamp. `parse_external` is the inverse.

    One stamp KEY, not one key per root. `STAMP_KEYS` is a fixed schema and
    `check` refuses a stamp missing any member of it; deriving the key set from
    the very declaration whose movement is under test would mean a dropped
    input takes its own coverage assertion with it, and "the field is gone"
    would read the same as "there was never such a field".
    """
    groups = external_input_groups(eng, crates, cargo_features)
    return " ".join(
        shlex.quote(f"{root_label}={external_group_digest(root_label, files)}")
        for root_label, files in groups
    )


def parse_external(value: str) -> dict[str, str]:
    """Split an `external=` stamp value into `{root label: digest}`.

    Inverse of `external_fingerprint`'s encoding. `rpartition`, not `partition`:
    a label may contain `=`, and the hex digest that follows never does.
    """
    entries: dict[str, str] = {}
    for token in shlex.split(value):
        label, sep, digest = token.rpartition("=")
        if sep:
            entries[label] = digest
    return entries


def external_diff(crate: str, recorded: str, expected: str) -> list[str]:
    """Name WHICH out-of-root input moved, rather than printing two digests.

    `external=` carries one digest per root precisely so this can be written.
    With a single digest there is nothing to say here beyond "something outside
    the repo changed" — a guard that fires without informing, which is the
    shape this field is structured to avoid.
    """
    was = parse_external(recorded)
    now = parse_external(expected)
    lines: list[str] = []
    for label in sorted(set(was) | set(now)):
        if label not in now:
            lines.append(
                f"{crate}: external: {label!r} is no longer a declared input, "
                f"but the artefact was built with it folded in"
            )
        elif label not in was:
            lines.append(
                f"{crate}: external: {label!r} is an input the artefact never "
                f"covered"
            )
        elif was[label] != now[label]:
            lines.append(f"{crate}: external: {label!r} changed")
    if not lines:
        # The packed values differ while every root agrees: an encoding this
        # engine did not write, or a reordering. Reporting nothing here would
        # turn a mismatch into a silent pass.
        lines.append(
            f"{crate}: external: {recorded!r} does not match {expected!r} "
            f"although every root agrees — the value was not written by this "
            f"engine"
        )
    return lines


def source_fingerprint(eng: Engine, crates: list[str], cargo_features: str) -> str:
    digest = hashlib.sha256()
    # Keep the cache/staleness key tied to extraction semantics rather than to
    # the implementation bytes of this driver. The driver-provided ABI covers
    # repo-specific extraction behaviour; the remaining fields automatically
    # cover the effective per-crate flags and target-layout configuration.
    config_fields = [
        ("fingerprint_schema", FINGERPRINT_SCHEMA),
        ("extraction_abi", eng.extraction_abi),
        ("cargo_features", cargo_features),
        ("layout_target_rustflags", eng.layout_target_rustflags),
    ]
    for crate in sorted(crates):
        spec = eng.spec(crate)
        flags = crate_flags(spec, cargo_features)
        config_fields.extend(
            [
                (f"{crate}.cargo_flags", "\x1f".join(flags)),
                (
                    f"{crate}.charon_flags",
                    "\x1f".join(charon_crate_flags(spec, cargo_features)),
                ),
                (
                    f"{crate}.layout_targets",
                    "\x1f".join(crate_layout_targets(eng, spec)),
                ),
                (
                    f"{crate}.layout_flags",
                    "\x1f".join(crate_layout_flags(spec, cargo_features, flags)),
                ),
            ]
        )
    for key, value in config_fields:
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for path in fingerprint_inputs(eng, crates, cargo_features):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        full_path = eng.root / path
        if full_path.is_file():
            digest.update(full_path.read_bytes())
        else:
            # The `--cached` half of the enumeration includes tracked paths
            # deleted in the working tree. A deletion is part of the source
            # state and must change the fingerprint instead of making
            # extraction unusable until commit.
            digest.update(b"<deleted>")
        digest.update(b"\0")
    return digest.hexdigest()


def prepend_msvc_link(env: dict[str, str]) -> None:
    if platform_info()[0] != "windows":
        return
    vswhere = Path(
        "C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if not vswhere.exists():
        print(
            "warn: vswhere not found; MSVC link.exe may be shadowed by Git's link.exe",
            file=sys.stderr,
        )
        return
    try:
        install = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        install = ""
    if not install:
        return
    msvc_root = Path(install) / "VC" / "Tools" / "MSVC"
    candidates = sorted(msvc_root.glob("*/bin/Hostx64/x64/link.exe"))
    if not candidates:
        candidates = sorted(msvc_root.glob("*/bin/hostx64/x64/link.exe"))
    if candidates:
        env["PATH"] = str(candidates[-1].parent) + os.pathsep + env.get("PATH", "")
    else:
        print(
            f"warn: MSVC link.exe not found under {install}; Git's link.exe may shadow it",
            file=sys.stderr,
        )


def charon_paths(charon_root: Path) -> tuple[str, Path, Path]:
    platform_key, charon_exe = platform_info()
    repo_parent = charon_root.parent
    shared = Path(os.environ.get("PYRE_SHARED_BUILD", repo_parent / ".pyre-build"))
    charon_dest = Path(
        os.environ.get("CHARON_DEST", shared / "charon" / platform_key)
    )
    return platform_key, charon_dest, charon_dest / charon_exe


def llbc_dest(out_dir: Path, root: Path) -> Path:
    dest = Path(os.environ.get("LLBC_DEST", out_dir))
    if not dest.is_absolute():
        dest = root / dest
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def charon_version(charon_dest: Path) -> str:
    stamp = charon_dest / ".installed-version"
    return stamp.read_text().strip() if stamp.exists() else "unknown"


def stamp_for(
    eng: Engine,
    *,
    crate: str,
    platform_key: str,
    charon_stamp: str,
    cargo_features: str,
    flags: list[str],
    charon_flags: list[str],
    layout_targets: list[str],
    layout_flags: list[str],
) -> str:
    return "\n".join(
        [
            f"crate={crate}",
            f"platform={platform_key}",
            f"charon={charon_stamp}",
            f"fingerprint_schema={FINGERPRINT_SCHEMA}",
            f"extraction_abi={eng.extraction_abi}",
            f"features={cargo_features}",
            f"flags={' '.join(flags)}",
            f"charon_flags={' '.join(charon_flags)}",
            f"layout_targets={' '.join(layout_targets)}",
            # The sidecar passes' own inputs: their cargo args and RUSTFLAGS
            # shape the extracted layouts, so changing either must re-extract.
            f"layout_flags={' '.join(layout_flags)}",
            f"layout_rustflags={eng.layout_target_rustflags}",
            f"source={source_fingerprint(eng, [crate], cargo_features)}",
            # Inputs `source=` structurally cannot cover: everything outside
            # this repo. Kept as its own field rather than folded into
            # `source=` so `check`'s per-field diff can say WHICH side moved —
            # "your own sources changed" and "a patched dependency in another
            # checkout changed" call for different actions, and collapsing
            # them would repeat the mistake this field exists to fix.
            f"external={external_fingerprint(eng, [crate], cargo_features)}",
        ]
    )


# Every field `stamp_for` writes, in order. A stamp that does not carry all of
# them was not written by this engine — or was truncated mid-write — and the
# comparison in `check` would then quietly skip whatever is absent.
STAMP_KEYS = (
    "crate",
    "platform",
    "charon",
    "features",
    "flags",
    "charon_flags",
    "layout_targets",
    "layout_flags",
    "layout_rustflags",
    "source",
    "external",
)


def parse_stamp(text: str) -> dict[str, str]:
    """Split a stamp into its `key=value` fields.

    Only used to replay `features=` and to render a readable per-field diff;
    the verdict in `check` is an exact text comparison, so a line this drops
    (blank, or without a `=`) still fails the check.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key] = value
    return fields


def file_mtime(path: Path) -> str:
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(
        timespec="seconds"
    )


def crate_layout_targets(eng: Engine, spec: CrateSpec) -> list[str]:
    """Cross targets this crate emits a layout sidecar for.

    The extraction host is never in the list: its layouts already live in
    the crate's own artefact.
    """
    extra = eng.layout_targets if spec.layout_targets is None else spec.layout_targets
    if not extra:
        return []
    host = host_triple(eng.root)
    return [t for t in extra if t != host]


def write_layout_sidecar(source: Path, dest: Path) -> None:
    """Reduce a full `.ullbc` to its type declarations.

    A cross-target extraction is only wanted for the struct layouts Charon
    resolved for that target; its function bodies are compiled under
    different `cfg`s and must not displace the host's. Dropping every
    declaration list but `type_decls` keeps the artefact a loadable
    `.ullbc` — the reader needs no special case, and the consumer merges
    it ahead of the host artefacts so its layouts win.

    Only the fields kept are decoded, not the whole file: bodies dominate
    the input (the interpreter's are ~94% of it), and parsing them costs an
    order of magnitude more memory than the text itself.
    """
    text = source.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()

    def field(name: str, *, want: type):
        key = f'"{name}":'
        at = text.find(key)
        if at < 0:
            raise SystemExit(f"extract-llbc.py: {source.name} has no `{name}` field")
        # `raw_decode` does not skip leading whitespace of its own.
        start = at + len(key)
        while text[start] in " \t\r\n":
            start += 1
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            # The substring scan is unscoped, so `"{name}":` could match inside
            # a string literal before the real key; then `start` points at
            # non-JSON and decoding fails.  Report it the same way the rest of
            # this file does instead of surfacing a bare traceback.
            raise SystemExit(
                f"extract-llbc.py: {source.name} `{name}` did not decode as JSON "
                f"({exc}); the reducer's key scan matched a non-field occurrence."
            ) from exc
        # Shape guard: a wrong (earlier) match can still decode as valid JSON of
        # the wrong type.  `type_decls` in particular must reach the sidecar
        # intact — it is the sole source of the target field offsets.
        if not isinstance(value, want):
            raise SystemExit(
                f"extract-llbc.py: {source.name} `{name}` decoded as "
                f"{type(value).__name__}, expected {want.__name__}; the reducer's "
                f"key scan matched a non-field occurrence."
            )
        return value

    slim = {
        "charon_version": field("charon_version", want=str),
        "has_errors": field("has_errors", want=bool),
        "translated": {
            "crate_name": field("crate_name", want=str),
            "type_decls": field("type_decls", want=list),
            "fun_decls": [],
        },
    }
    del text
    dest.write_text(json.dumps(slim), encoding="utf-8")


def ensure_charon_std(charon_bin: Path, targets: list[str], root: Path) -> None:
    """Install the target `std` Charon's pinned toolchain needs.

    Charon drives its own nightly rustc, so a target installed for the
    build toolchain is not necessarily installed for Charon's. Adding it is
    idempotent and fast when already present.
    """
    toolchain_name = Path(run_capture([str(charon_bin), "toolchain-path"], cwd=root).strip()).name
    for target in targets:
        command = ["rustup", "target", "add", target, "--toolchain", toolchain_name]
        if subprocess.run(command).returncode != 0:
            raise SystemExit(
                f"extract-llbc.py: could not install '{target}' std for Charon's "
                f"toolchain '{toolchain_name}'.\n"
                f"  run: {' '.join(command)}\n"
                "  or drop the target from LLBC_LAYOUT_TARGETS to extract host "
                "layouts only (a cross-target build then reads the host's field "
                "offsets)."
            )


def extract(eng: Engine, args: argparse.Namespace) -> None:
    cargo_features = eng.cargo_features
    platform_key, charon_dest, charon_bin = charon_paths(eng.charon_root)

    if not charon_bin.exists():
        raise SystemExit(
            f"extract-llbc.py: charon not installed at {charon_bin}\n"
            "  run: scripts/install-charon.py"
        )

    dest_dir = llbc_dest(eng.out_dir, eng.root)
    charon_stamp = charon_version(charon_dest)
    unstamped: list[str] = []
    env = os.environ.copy()
    prepend_msvc_link(env)

    crate_attr = "-Zcrate-attr=feature(cfg_select)"
    env["RUSTC_BOOTSTRAP"] = "1"
    env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " " + crate_attr).strip()
    # Charon reads MIR straight from rustc; the compiled binary is discarded
    # and only the `.ullbc` is kept, so debuginfo is dead weight here. Drop it
    # to skip DWARF generation across the whole extraction graph. The nightly
    # extraction build fingerprints separately from the stable build (distinct
    # rustc), so this never thrashes the runtime build's cache, and the LLBC is
    # independent of debuginfo so the artefact is byte-identical.
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "0")
    # The dev profile enables incremental compilation, which splits a crate into
    # codegen units and reuses object files from the previous session. Charon
    # drives rustc for the crates it extracts while plain rustc builds the rest,
    # and the two disagree about which CGUs are still valid: the resulting
    # archive fuses objects from different sessions, so CGU-local symbols
    # (`...drop_in_place...llvm.<hash>`) are referenced by one object and
    # defined in none. The failure surfaces far from its cause, as
    # `Undefined symbols for architecture arm64` while LINKING the
    # `pyre-jit-trace` BUILD SCRIPT, which is a host binary and therefore the
    # first thing in the graph that has to resolve real code out of
    # `libmajit_translate`. The tell is that the `.rcgu.o` files named in the
    # linker error carry a different session tag than the loose ones on disk.
    # Extraction only needs MIR, so incremental buys nothing here.
    env["CARGO_INCREMENTAL"] = "0"
    # Dependency build scripts run while Charon extracts a target crate. They
    # must not recursively demand the very LLBC artefact currently being
    # produced (pyre-jit -> pyre-jit-trace -> pyre-jit.ullbc). Consumers may
    # use this explicit extraction mode to emit compile-only placeholders;
    # `rerun-if-env-changed` ensures the following normal build regenerates
    # production artifacts from the completed LLBC set.
    env["MAJIT_LLBC_EXTRACTION"] = "1"
    host_config_env = {
        "CARGO_UNSTABLE_HOST_CONFIG": "true",
        "CARGO_UNSTABLE_TARGET_APPLIES_TO_HOST": "true",
    }
    host_config = [
        "--config",
        "target-applies-to-host=false",
        "--config",
        f'host.rustflags=["{crate_attr}"]',
    ]

    prepared_std: set[str] = set()

    for crate in args.crates or eng.default_crates:
        spec = eng.spec(crate)
        path = spec.crate_dir
        flags = crate_flags(spec, cargo_features)
        charon_flags = charon_crate_flags(spec, cargo_features)
        if not path.is_dir():
            raise SystemExit(f"extract-llbc.py: missing crate dir for '{crate}' at {path}")

        dest = dest_dir / spec.output_name
        stamp_path = dest.with_suffix(dest.suffix + ".fingerprint")
        layout_flags = crate_layout_flags(spec, cargo_features, flags)
        stamp = stamp_for(
            eng,
            crate=crate,
            platform_key=platform_key,
            charon_stamp=charon_stamp,
            cargo_features=cargo_features,
            flags=flags,
            charon_flags=charon_flags,
            layout_targets=crate_layout_targets(eng, spec),
            layout_flags=layout_flags,
        )

        sidecars = [
            dest_dir / spec.layout_sidecar_name(t) for t in crate_layout_targets(eng, spec)
        ]
        if (
            not args.force
            and dest.exists()
            and dest.stat().st_size > 0
            and all(s.exists() and s.stat().st_size > 0 for s in sidecars)
            and stamp_path.exists()
            and stamp_path.read_text() == stamp + "\n"
        ):
            print(f"=== skipping {crate} -> {dest} (fingerprint unchanged) ===")
            continue

        print(f"=== extracting {crate} -> {dest} ===")
        # Charon writes the `.ullbc` only while rustc actually compiles
        # the crate. Once the fingerprint skip above is past, the artefact
        # is known absent or stale and must be (re)generated — but a warm
        # `target/<host-triple>/` cache (e.g. `build/` was wiped while the
        # build cache survived) makes the inner `cargo build` skip rustc
        # and emit nothing, leaving `dest` missing. Touch the crate root
        # to dirty just this unit's fingerprint so it always recompiles
        # and re-emits; dependency crates stay cached (their MIR reaches
        # Charon via rlib metadata), so re-runs remain cheap.
        crate_root = path / "src" / "lib.rs"
        if not crate_root.exists():
            crate_root = path / "src" / "main.rs"
        crate_root.touch()

        host_env = {**env, **host_config_env}
        command = [
            str(charon_bin),
            "cargo",
            "--ullbc",
            "--dest-file",
            str(dest),
            *charon_flags,
            "--",
            *flags,
            *host_config,
        ]
        subprocess.run(command, cwd=path, env=host_env, check=True)
        # Fail loud rather than letting a missing artefact surface later
        # as an opaque build.rs panic ("build/llbc/ is missing …").
        if not dest.exists() or dest.stat().st_size == 0:
            raise SystemExit(
                f"extract-llbc.py: Charon emitted no artefact at {dest}\n"
                "  the crate compiled but produced no MIR — "
                "inspect the Charon output above"
            )

        # One extra extraction per cross target, reduced to its type
        # declarations. Charon's own `--targets` aggregation would fold
        # every target's layouts into this one artefact, but it aggregates
        # the *bodies* too: a `cfg`-differing function lands two or three
        # times over, and a portal that must resolve to an exact graph no
        # longer does. Extracting each target separately keeps the host
        # artefact's bodies untouched, and dropping everything but
        # `type_decls` from the cross-target one leaves only what a
        # cross-target build actually lacks — its own field offsets.
        for target in crate_layout_targets(eng, spec):
            if target not in prepared_std:
                ensure_charon_std(charon_bin, [target], eng.root)
                prepared_std.add(target)
            sidecar = dest_dir / spec.layout_sidecar_name(target)
            print(f"=== extracting {crate} layouts for {target} -> {sidecar} ===")
            layout_env = dict(env)
            if eng.layout_target_rustflags:
                layout_env["RUSTFLAGS"] = (
                    layout_env.get("RUSTFLAGS", "") + " " + eng.layout_target_rustflags
                ).strip()
            full = sidecar.with_suffix(sidecar.suffix + ".full")
            crate_root.touch()
            subprocess.run(
                [
                    str(charon_bin),
                    "cargo",
                    "--ullbc",
                    "--dest-file",
                    str(full),
                    "--targets",
                    target,
                    *charon_flags,
                    "--",
                    *layout_flags,
                ],
                cwd=path,
                env=layout_env,
                check=True,
            )
            if not full.exists() or full.stat().st_size == 0:
                raise SystemExit(
                    f"extract-llbc.py: Charon emitted no {target} artefact at {full}"
                )
            write_layout_sidecar(full, sidecar)
            full.unlink()
            print(f"    wrote {sidecar} ({sidecar.stat().st_size} bytes)")
        # Guard the fingerprint exclusion (CrateSpec.excluded_deps): a package
        # dropped from this crate's fingerprint must not appear in its artefact,
        # else a later edit to that package would silently serve a stale cache.
        artefact_bytes = dest.read_bytes()
        for pkg in spec.excluded_deps:
            symbol = pkg.replace("-", "_").encode("utf-8")
            if symbol in artefact_bytes:
                raise SystemExit(
                    f"extract-llbc.py: {dest.name} references '{pkg}', which is "
                    f"excluded from its fingerprint.\n"
                    f"  Remove '{pkg}' from the '{crate}' spec's excluded_deps"
                    f" — its source now affects this artefact, so the artefact"
                    f" must re-extract when it changes."
                )
        # `stamp` was computed at the top of this iteration, BEFORE the charon
        # build that has just run for minutes. If the tree moved in between it
        # names a source hash this artefact was not built from, and the wrong
        # direction is a RETURN to the stamped state — a reverted edit, a branch
        # switched back — where `check` then reports FRESH over an artefact
        # built from other sources. Re-stamping with the post-build value would
        # be equally untrue: the artefact straddles both trees and belongs to
        # neither, and a stamp that looks authoritative is worse than none.
        if source_fingerprint(eng, [crate], cargo_features) != parse_stamp(stamp)["source"]:
            stamp_path.unlink(missing_ok=True)
            unstamped.append(crate)
            print(
                f"    REFUSING to stamp {dest.name}: the tree moved during its"
                f" build, so no source hash describes this artefact. Left"
                f" unstamped — reported as freshness UNKNOWN, not as fresh."
            )
            continue
        stamp_path.write_text(stamp + "\n")
        print(f"    wrote {dest} ({dest.stat().st_size} bytes)")

    print()
    if unstamped:
        raise SystemExit(
            "extract-llbc.py: the tree moved while extracting "
            + ", ".join(unstamped)
            + ".\n  Those artefacts are written but deliberately unstamped."
            "\n  Re-run with --force once the tree is quiet."
        )
    print(f"all extractions complete. artefacts under: {dest_dir}")


def check(eng: Engine, args: argparse.Namespace) -> None:
    """Refuse an artefact whose stamp does not match the tree it sits beside.

    `extract` already computes this comparison — it is the `skipping <crate>
    (fingerprint unchanged)` test — but only a caller who runs the extractor
    ever benefits from it. A consumer that opens `<out_dir>/<crate>.ullbc` as
    found gets whatever was last written there, and a stale artefact does not
    fail: it answers, in detail, about sources that no longer exist. So this is
    the same comparison, run by the consumer, reported through the exit status.
    Nonzero means DO NOT READ the artefact.

    Every way of reaching the end WITHOUT having compared the stamp against a
    freshly computed one is a refusal, not a pass: an absent stamp, an empty
    stamp, a stamp missing fields. Treating "nothing to compare" as "nothing
    wrong" is what makes a gate decorative — a harness reads the exit status,
    not the output.

    Two things are deliberately not gated, and one is simply not covered here
    — the list is written out because an incomplete "what this does not check"
    reads as a complete one, which is the same shape as an allow-list that
    silently omits a member:

      * `features=` is replayed out of the stamp rather than compared. The
        stamp records the configuration the artefact was built under, and the
        question here is whether the artefact is current FOR that
        configuration; comparing it against this process's `CARGO_FEATURES`
        would refuse a byte-correct artefact whenever the caller left the
        default in place. Replaying keeps `flags=`, `charon_flags=`,
        `layout_flags=`, `source=` and `external=` real comparisons — all
        five are recomputed from the replayed value. (`external=` belongs on
        that list because the dependency walk it feeds off is itself run
        per feature set, so a feature that pulls in a patched path dep
        changes which out-of-root inputs exist.)
      * artefact and stamp mtimes are printed, never gated. `extract` writes
        both in one pass, and the source digest already answers the question a
        timestamp only approximates.
      * the `excluded_deps` exclusion is not re-asked here, and MOSTLY DOES NOT
        NEED TO BE — do not add a second checker without reading this first.
        `extract` re-reads the artefact and refuses if a NAMED excluded
        package is referenced by it; `stamp_path.write_text` is the
        stamp's ONLY writer and sits immediately after that guard, so a
        violation raises before any stamp exists. A matching stamp is therefore
        the guard's certificate: it could only have been written by an
        extraction the guard passed. The skip path inherits this — it fires on
        `stamp == recorded`, and that recorded stamp had the same provenance.
        The oracle discriminates rather than being vacuous: `majit_translate`
        occurs 373 times in `pyre-jit.ullbc`, which excludes nothing, and 0
        times in `pyre-interpreter.ullbc`, which excludes it.

        Three narrow things the certificate does NOT carry:

          - it does not distinguish "the guard passed" from "the guard was
            vacuous". A spec with empty `excluded_deps` runs an empty loop and
            writes an indistinguishable stamp.
          - it does not cover the EXCLUSIVELY-REACHED SUBTREE the exclusion
            also drops, and ⛔ widening the loop to cover it would be worse
            than leaving it uncovered, because the guard's power is PER
            SYMBOL. `majit_charon_reader` — dropped as `majit-translate`'s
            subtree — occurs 0 times in ALL SIX artefacts this tree builds,
            `pyre-jit.ullbc` included, so no artefact can serve as its
            positive control and a widened guard would pass without
            evidence. What actually certifies the subtree is INHERITANCE:
            the walk drops a package only when EVERY path to it runs through
            an excluded one, and the named package's absence is checked
            here, so a package that cannot appear without it cannot appear.
            `pyre-object` is NOT a second witness for that absence — it does
            not depend on `majit-translate` at all, so its exclusion is an
            inert declaration and its 0 is uninformative.
          - `--force` re-extracts with the skip bypassed, and an excluded
            package's sources changing is BY CONSTRUCTION invisible to
            `source=`. So a forced run can write a violating artefact, raise at
            the guard, and leave the previous stamp in place still matching the
            tree — after which this function passes. It fails loud once, at the
            forced extraction, and is silent on every check after. (Read from
            the code path, not reproduced: reproducing it needs a real
            extraction. The half that IS demonstrated is that `check` never
            looks at artefact content — a stamp-matching fixture holding
            arbitrary bytes passes.)

    Nothing here re-extracts. Re-extraction runs a whole-crate Charon build and
    writes into the working tree, so it stays a human's scheduling decision and
    the refusal names the command instead. No cost figure is quoted anywhere in
    this function: what it costs depends on how warm the cargo cache is, and a
    number carried over from one cold run reads as measured when it is not.
    `LLBC_DEST` points the check at a directory other than the driver's default,
    which is what a caller who must not disturb the live artefacts uses.

    Exit 0 has to be EARNED, never fallen into. `verified` records the crates
    that positively matched, and the epilogue refuses unless every requested
    crate is in it — so an empty crate list, or any future branch that forgets
    to file a failure, refuses instead of reporting success by default.
    """
    platform_key, charon_dest, _ = charon_paths(eng.charon_root)
    charon_stamp = charon_version(charon_dest)
    dest_dir = llbc_dest(eng.out_dir, eng.root)
    # ⚠ `crates` is what was REQUESTED, and the epilogue's `unaccounted` check
    # depends on it staying that way. Rebuilding this list from resolved specs
    # would drop a name the loop could not process before the epilogue ever saw
    # it, turning "this crate was never confirmed" back into silent success.
    crates = args.crates or eng.default_crates

    stale: list[str] = []
    verified: set[str] = set()
    # Feature set to re-extract each crate under: the one its stamp records, so
    # a mixed set does not collapse into one wrong remedy command. A crate whose
    # stamp never got read keeps this process's feature set.
    stamped_features: dict[str, str] = {crate: eng.cargo_features for crate in crates}

    for crate in crates:
        spec = eng.spec(crate)
        dest = dest_dir / spec.output_name
        stamp_path = dest.with_suffix(dest.suffix + ".fingerprint")
        print(f"=== checking {crate} -> {dest} ===")

        if not dest.exists():
            stale.append(f"{crate}: no artefact at {dest}")
            continue
        if dest.stat().st_size == 0:
            stale.append(
                f"{crate}: artefact {dest} is 0 bytes — Charon aborted before "
                f"writing it"
            )
            continue
        print(f"    artefact {dest.stat().st_size} bytes, mtime {file_mtime(dest)}")

        for target in crate_layout_targets(eng, spec):
            sidecar = dest_dir / spec.layout_sidecar_name(target)
            if not sidecar.exists() or sidecar.stat().st_size == 0:
                stale.append(
                    f"{crate}: {target} layout sidecar {sidecar} is missing or "
                    f"empty — field offsets for that target would be read out "
                    f"of the extraction host's layouts"
                )

        if not stamp_path.exists():
            stale.append(
                f"{crate}: no fingerprint stamp at {stamp_path} — the artefact "
                f"records nothing about the sources it came from, so there is "
                f"nothing to compare it against"
            )
            continue
        stamp_bytes = stamp_path.read_bytes()
        if not stamp_bytes:
            stale.append(
                f"{crate}: fingerprint stamp {stamp_path} is 0 bytes — it "
                f"records no source digest, so it cannot match the tree"
            )
            continue
        text = stamp_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            stale.append(
                f"{crate}: fingerprint stamp {stamp_path} holds only whitespace"
            )
            continue
        print(f"    stamp {len(stamp_bytes)} bytes, mtime {file_mtime(stamp_path)}")

        recorded = parse_stamp(text)
        missing = [key for key in STAMP_KEYS if key not in recorded]
        if missing:
            # Order matters: the LIKELIEST cause is listed first, and it is the
            # one the two original guesses both excluded. Adding a key to
            # STAMP_KEYS makes EVERY stamp already on disk land here at once,
            # so right after such a change this arm fires for a reason that is
            # neither "a foreign tool wrote it" nor "it was truncated" — the
            # stamp is this engine's own output from before the field existed.
            # Saying only those two sends a reader hunting a corruption that is
            # not there. Measured after `external=` was added: all five live
            # pyre stamps refuse through here, none of them damaged.
            stale.append(
                f"{crate}: fingerprint stamp {stamp_path} has no "
                f"{', '.join(missing)} field — written before this engine "
                f"recorded that field, or by another tool, or truncated. "
                f"Re-extract either way; the artefact predates what the stamp "
                f"is now required to cover, so it cannot be compared"
            )
            continue

        features = recorded["features"]
        stamped_features[crate] = features
        flags = crate_flags(spec, features)
        expected = stamp_for(
            eng,
            crate=crate,
            platform_key=platform_key,
            charon_stamp=charon_stamp,
            cargo_features=features,
            flags=flags,
            charon_flags=charon_crate_flags(spec, features),
            layout_targets=crate_layout_targets(eng, spec),
            layout_flags=crate_layout_flags(spec, features, flags),
        )
        if text == expected + "\n":
            print(f"    fingerprint matches the tree (source={recorded['source']})")
            verified.add(crate)
            continue

        want = parse_stamp(expected)
        differing = [key for key in STAMP_KEYS if recorded[key] != want.get(key)]
        if not differing:
            # The text differs while every modelled field agrees, so the stamp
            # carries content this engine does not write. Reporting nothing
            # here would turn a mismatch into a silent pass.
            stale.append(
                f"{crate}: fingerprint stamp {stamp_path} does not match "
                f"byte-for-byte although every field agrees — it carries "
                f"content this engine did not write"
            )
            continue
        for key in differing:
            if key == "external":
                # Two packed per-root lists, not two hashes: printing them as
                # opaque values would hand the reader kilobytes and no answer.
                stale.extend(
                    external_diff(crate, recorded[key], want.get(key, ""))
                )
                continue
            stale.append(
                f"{crate}: {key}: artefact says {recorded[key]!r}, "
                f"tree says {want.get(key)!r}"
            )

    # Exit 0 requires a positive match for every requested crate, and at least
    # one crate to have been requested. Both halves matter: an empty list has no
    # unaccounted crates, so testing only for those would let "checked nothing"
    # through as success — the exact shape this whole function exists to refuse.
    unaccounted = [crate for crate in crates if crate not in verified]
    if not stale and (not crates or unaccounted):
        stale.append(
            "checked nothing for: " + (" ".join(unaccounted) or "(no crates requested)")
        )
    if not stale:
        print()
        print(f"llbc artefacts current: {' '.join(crates)}")
        return

    driver = sys.argv[0] or "scripts/extract-llbc.py"
    # The per-crate lines above went to stdout, which is block-buffered when
    # this runs under a harness; flush so the verdict does not land before the
    # evidence it was drawn from.
    sys.stdout.flush()
    print(file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("LLBC STALE — do not read these artefacts", file=sys.stderr)
    for line in stale:
        print(f"  {line}", file=sys.stderr)
    print(
        "\n"
        "  Re-extract before reading. This runs a whole-crate Charon build and\n"
        "  writes into the working tree, so nothing here runs it for you —\n"
        "  how long it takes depends on how warm the cargo cache is:",
        file=sys.stderr,
    )
    for features in dict.fromkeys(stamped_features.values()):
        group = [c for c in crates if stamped_features[c] == features]
        print(
            f"      CARGO_FEATURES={features} python3 {driver} --force "
            f"{' '.join(group)}",
            file=sys.stderr,
        )
    print("=" * 72, file=sys.stderr)
    raise SystemExit(1)


def self_test_exclusion_diamond() -> None:
    """Drive `_package_closure` on a synthetic diamond; check BOTH its halves.

    `excluded_deps` has to do two opposite things at once, and a real tree
    cannot demonstrate either. Excluding a package must drop what only that
    package reaches, and must NOT drop what something else also reaches — and a
    fingerprint that gets this wrong still hashes to something, still compares
    equal to itself, and still passes `--check`. The pre-fix engine filtered at
    emission and carried four unreferenced files for 30 days without a single
    red run. So the property is checked here on a graph built for it.

    The graph, with `excluded` excluded:

        root ──► keep_via_both ──► shared
          └────► excluded ──┬────► shared           (also reached from above)
                            └────► behind_excluded  (reached ONLY via excluded)

    Two-sided, and neither side alone is sufficient:

    * `behind_excluded` must be ABSENT. An engine that filters at emission
      keeps it — that is the #152 defect, and this is the only assertion that
      sees it.
    * `shared` must be PRESENT. An engine that prunes by ancestry instead
      (dropping everything downstream of an excluded node) also passes the
      first assertion, and drops a package the tree genuinely depends on. That
      is a wrong answer in the unsafe direction — an under-fingerprinted crate
      whose artefact goes stale silently.

    The unexcluded control runs first and is not a formality: it establishes
    that both nodes are reachable AT ALL. Without it, "absent" and "present"
    are being read off a graph that might reach neither, and the excluded run
    would pass for a reason unrelated to exclusion.

    ⚠ Not covered here: the `dep_kinds`/dev-edge filter in the same walk. It is
    a different property with a different failure mode and no fixture yet.
    """
    def pkg(name, source=None):
        return {"id": f"{name} 0.0.0", "name": name, "source": source}

    def node(name, *deps):
        return (
            f"{name} 0.0.0",
            {"deps": [{"pkg": f"{d} 0.0.0", "dep_kinds": []} for d in deps]},
        )

    names = ["root", "keep_via_both", "excluded", "shared", "behind_excluded"]
    by_id = {p["id"]: p for p in (pkg(n) for n in names)}
    resolve_nodes = dict(
        [
            node("root", "keep_via_both", "excluded"),
            node("keep_via_both", "shared"),
            node("excluded", "shared", "behind_excluded"),
            node("shared"),
            node("behind_excluded"),
        ]
    )
    root_ids = ["root 0.0.0"]

    def walk(exclude):
        return sorted(
            p["name"] for p in _package_closure(root_ids, by_id, resolve_nodes, exclude)
        )

    control = walk(set())
    excluded = walk({"excluded"})
    print("  exclusion diamond")
    print(f"    exclude {{}}            {control}")
    print(f"    exclude {{excluded}}    {excluded}")

    failures = []
    # The control first: an unreachable node proves nothing by being absent.
    for name in ("shared", "behind_excluded"):
        if name not in control:
            failures.append(
                f"control walk does not reach {name!r} — the fixture graph is "
                f"broken, and every verdict below it is vacuous"
            )
    if not failures:
        if "behind_excluded" in excluded:
            failures.append(
                "'behind_excluded' survived the exclusion — the exclusion is "
                "being applied at emission rather than in the walk, so a "
                "package reachable ONLY through an excluded one still moves "
                "the fingerprint (the #152 defect)"
            )
        if "shared" not in excluded:
            failures.append(
                "'shared' was dropped by the exclusion — the walk is pruning "
                "by ancestry rather than by reachability-avoiding-the-excluded, "
                "so a package the tree still depends on left the fingerprint "
                "and its artefact can go stale unnoticed"
            )
        if "excluded" in excluded:
            failures.append("the excluded package itself is in the closure")
    if failures:
        sys.stdout.flush()  # as in check(): the verdict must not outrun its evidence
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        raise SystemExit(1)


def self_test(eng: Engine, crates: list[str], cargo_features: str) -> None:
    """A/B/A the fingerprint against a new untracked `.rs` under a covered path.

    The blind spot being guarded is silent by construction: `git ls-files` lists
    tracked paths only, so before `fingerprint_inputs` folded in `--others
    --exclude-standard`, adding a brand-new module left the hash unmoved and
    `--check` blessed an artefact built from different source. A wrong answer
    there prints nothing, so the guard needs a demonstration rather than a
    reading of the patch. Both legs are required: that the hash *moves* for the
    new file, and that removing the probe *restores* it, so a change which
    invalidated every stamp cannot pass by refusing everything.

    The probe exists for one `source_fingerprint` call and is removed in a
    `finally`. On a shared worktree that is a real if brief mutation — anyone
    fingerprinting the same crate in that instant sees the moved hash.
    """
    inputs = fingerprint_inputs(eng, crates, cargo_features)
    # Deepest `.rs`: exact-file pathspecs (Cargo.toml, target roots, build.rs)
    # sit shallow, so a deeply nested file was listed by a `dir/` pathspec and
    # its directory is therefore covered for new files too.
    covered = max((p for p in inputs if p.suffix == ".rs"), key=lambda p: (len(p.parts), p.as_posix()), default=None)
    if covered is None:
        raise SystemExit(f"self-test: no .rs input for {' '.join(crates)}")
    probe = eng.root / covered.parent / "__llbc_self_test_probe.rs"
    if probe.exists():
        raise SystemExit(f"self-test: {probe} already exists; refusing to clobber it")
    before = source_fingerprint(eng, crates, cargo_features)
    try:
        probe.write_text("// transient probe written by --self-test; safe to delete\n")
        moved = source_fingerprint(eng, crates, cargo_features)
        listed = probe.relative_to(eng.root) in set(fingerprint_inputs(eng, crates, cargo_features))
    finally:
        probe.unlink(missing_ok=True)
    after = source_fingerprint(eng, crates, cargo_features)
    print(f"    probe          {probe.relative_to(eng.root).as_posix()}")
    print(f"    before         {before}")
    print(f"    with probe     {moved}  (listed as an input: {listed})")
    print(f"    after removal  {after}")
    failures = []
    if moved == before:
        failures.append("a new untracked .rs did not move the fingerprint — the guard is blind")
    if after != before:
        failures.append("removing the probe did not restore the fingerprint — it left residue")
    if failures:
        sys.stdout.flush()  # as in check(): the verdict must not outrun its evidence
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nself-test passed: {' '.join(crates)}")


def run_cli(
    specs: dict[str, CrateSpec],
    default_crates: list[str],
    *,
    root: Path,
    out_dir: Path,
    extraction_abi: str,
    base_pathspecs: list[str] | None = None,
    charon_root: Path | None = None,
    metadata_feature_crates: tuple[str, ...] = (),
    layout_targets: tuple[str, ...] = (),
    layout_target_rustflags: str = "",
    external_inputs: tuple[Path, ...] = (),
) -> None:
    """Argparse UX shared by every driver (positional crates, --force, …)."""
    all_crates = " ".join(specs)
    parser = argparse.ArgumentParser(
        description="Extract Charon ULLBC artefacts with source-fingerprint skip logic."
    )
    parser.add_argument("crates", nargs="*", help=f"known: {all_crates}")
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--list-inputs", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the existing artefacts against the tree and exit nonzero "
        "if any is stale; never extracts",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove the source fingerprint still sees a new untracked .rs; "
        "writes and removes one transient probe file, never extracts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.environ.get("LLBC_FORCE_REEXTRACT") == "1",
    )
    args = parser.parse_args()

    # Default feature set is `dynasm`, matching the default JIT backend. The
    # ULLBC that feeds trace codegen is backend-agnostic — `dynasm` and
    # `cranelift` extraction yield byte-identical generated code — so the
    # lighter backend skips compiling the cranelift-codegen tree (~33 crates)
    # the dynasm build never needs. A driver whose crates ignore `{features}`
    # is unaffected by this default. Override with `CARGO_FEATURES`.
    cargo_features = os.environ.get("CARGO_FEATURES", "dynasm")
    # `LLBC_LAYOUT_TARGETS` overrides the driver's cross-target layout set
    # (comma-separated); the empty string extracts host layouts only.
    layout_override = os.environ.get("LLBC_LAYOUT_TARGETS")
    if layout_override is not None:
        layout_targets = tuple(t.strip() for t in layout_override.split(",") if t.strip())
    eng = Engine(
        specs=specs,
        default_crates=default_crates,
        root=root,
        out_dir=out_dir,
        base_pathspecs=list(base_pathspecs) if base_pathspecs else ["Cargo.lock", "Cargo.toml"],
        charon_root=charon_root or root,
        cargo_features=cargo_features,
        extraction_abi=extraction_abi,
        metadata_feature_crates=metadata_feature_crates,
        layout_targets=layout_targets,
        layout_target_rustflags=layout_target_rustflags,
        external_inputs=tuple(external_inputs),
    )
    # Before ANY subcommand, including the read-only instruments: an inert
    # pathspec makes `--list-inputs` and `--fingerprint` report a coverage the
    # stamp does not have, and those are the two outputs a reader trusts when
    # deciding whether the guard is working.
    refuse_inert_pathspecs(eng)

    crates = args.crates or default_crates
    if args.list_inputs:
        for path in fingerprint_inputs(eng, crates, cargo_features):
            print(path.as_posix())
        # Marked, because the two channels are hashed into different stamp
        # fields and an unmarked union would read as one list of repo-relative
        # paths — which is what made the out-of-root inputs easy to miss. The
        # root is named on every line because it, not the file, is the unit
        # `external=` records a digest for and `check` reports a move against.
        for root_label, files in external_input_groups(eng, crates, cargo_features):
            if not files:
                # A declared root the filter emptied. It still holds a digest,
                # so listing nothing for it would under-report the coverage.
                print(f"external[{root_label}]: (no files)")
            for label, _ in files:
                print(f"external[{root_label}]:{label}")
        return
    if args.fingerprint:
        # BOTH fields. Printing only `source=` here previously cost a session:
        # three legs returned identical hashes that were consistent with every
        # hypothesis, because the field under test was not in the output.
        print(f"source={source_fingerprint(eng, crates, cargo_features)}")
        print(f"external={external_fingerprint(eng, crates, cargo_features)}")
        return
    if args.self_test:
        # The diamond runs first because it mutates nothing: on a shared
        # worktree, a broken input-set computation should be reported without
        # writing a probe file into the tree to find it out. It is also the
        # case `self_test` structurally cannot cover — that one compares the
        # fingerprint against itself, so it stays green over an input set that
        # is silently wrong.
        self_test_exclusion_diamond()
        self_test(eng, crates, cargo_features)
        return
    if args.check:
        check(eng, args)
        return
    extract(eng, args)


# This module is the shared extraction ENGINE. Every consumer imports it — the
# per-repo wrappers call `run_cli` with their own `SPECS` — and nothing has ever
# invoked it as a program.
#
# Without this block, running it directly defined `main()` and fell off the end:
# zero output, **exit 0**, for every argument including `--check`, `--fingerprint`
# and `--help`. A freshness check aimed here reported success without executing,
# which is worse than the staleness it was run to detect.
#
# The body refuses instead of dispatching. Adding a `main()` call would make
# direct invocation *work* and mint a second, under-specified entry point: the
# engine alone cannot know which crates to extract or where their sources live —
# that is exactly what a wrapper's `CrateSpec` and `PYRE_ROOT` resolution supply.
if __name__ == "__main__":
    raise SystemExit(
        "llbc_extract.py is the shared extraction ENGINE, not an entry point.\n"
        "  pyre crates:  python3 scripts/extract-llbc.py ...\n"
        "  cel crates:   python3 cel-jit/scripts/extract-llbc.py ...\n"
        "Those wrappers supply the CrateSpec table this module needs."
    )
