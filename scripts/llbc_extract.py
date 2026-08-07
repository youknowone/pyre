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


def fingerprint_inputs(eng: Engine, crates: list[str], cargo_features: str) -> list[Path]:
    root = eng.root
    target_names: list[str] = []
    pathspecs = list(eng.base_pathspecs)

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

            seen: set[str] = set()
            stack = [by_name[name]["id"] for name in target_names]
            closure = []
            while stack:
                package_id = stack.pop()
                if package_id in seen:
                    continue
                seen.add(package_id)
                package = by_id[package_id]
                # Apply the exclusion in the walk, not after it. This drops a
                # package's exclusively-reached subtree as well as the named
                # package. A node also reachable from a non-excluded parent is
                # still pushed along that path and therefore remains present.
                # The extracted artefact guard certifies the named package's
                # absence; a subtree that has no path avoiding that package
                # cannot occur without it.
                if package["name"] in exclude:
                    continue
                closure.append(package)

                for dep in resolve_nodes.get(package_id, {}).get("deps", []):
                    dep_kinds = dep.get("dep_kinds", [])
                    # An empty `dep_kinds` is a normal (non-dev) edge; only
                    # drop deps whose every listed kind is `dev`.
                    if dep_kinds and all(
                        kind.get("kind") == "dev" for kind in dep_kinds
                    ):
                        continue
                    dep_package = by_id.get(dep["pkg"])
                    if dep_package is not None and dep_package.get("source") is None:
                        stack.append(dep_package["id"])

            for package in closure:
                package_dir = Path(package["manifest_path"]).resolve().parent
                if package_dir.is_relative_to(root):
                    rel_dir = package_dir.relative_to(root).as_posix()
                    pathspecs.append(f"{rel_dir}/Cargo.toml")
                for target in package["targets"]:
                    kinds = set(target["kind"])
                    if not ({"lib", "bin", "custom-build"} & kinds):
                        continue
                    src_path = Path(target["src_path"]).resolve()
                    if src_path.is_relative_to(root):
                        rel_src = src_path.relative_to(root).as_posix()
                        pathspecs.append(rel_src)
                        if "custom-build" not in kinds:
                            pathspecs.append(str(Path(rel_src).parent) + "/")

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
    return sorted(files, key=lambda path: path.as_posix())


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
        stamp_path.write_text(stamp + "\n")
        print(f"    wrote {dest} ({dest.stat().st_size} bytes)")

    print()
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

    Two things are deliberately not gated:

      * `features=` is replayed out of the stamp rather than compared. The
        stamp records the configuration the artefact was built under, and the
        question here is whether the artefact is current FOR that
        configuration; comparing it against this process's `CARGO_FEATURES`
        would refuse a byte-correct artefact whenever the caller left the
        default in place. Replaying keeps `flags=`, `charon_flags=`,
        `layout_flags=` and `source=` real comparisons — all four are
        recomputed from the replayed value.
      * artefact and stamp mtimes are printed, never gated. `extract` writes
        both in one pass, and the source digest already answers the question a
        timestamp only approximates.

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
            stale.append(
                f"{crate}: fingerprint stamp {stamp_path} has no "
                f"{', '.join(missing)} field — it was not written by this "
                f"engine, or was truncated"
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
    )

    crates = args.crates or default_crates
    if args.list_inputs:
        for path in fingerprint_inputs(eng, crates, cargo_features):
            print(path.as_posix())
        return
    if args.fingerprint:
        print(source_fingerprint(eng, crates, cargo_features))
        return
    if args.check:
        check(eng, args)
        return
    extract(eng, args)
