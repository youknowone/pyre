"""Charon ULLBC extraction engine with source-fingerprint skip logic.

Import-only library: a per-repo driver declares its crate table as a dict of
`CrateSpec`s and calls `run_cli(...)`. This module carries ZERO crate names —
every crate (its dir, cargo flags, fingerprint inputs, output artefact) is
declared by the driver, so the engine stays neutral about which consumer
(pyre, or an external interpreter crate) it is extracting for.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


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


def run_capture(args: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ).stdout


def host_triple(root: Path) -> str:
    if value := os.environ.get("LLBC_TARGET_TRIPLE"):
        return value
    rustc_info = run_capture(["rustc", "-vV"], cwd=root)
    for line in rustc_info.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ")
    raise SystemExit("extract-llbc.py: could not determine rustc host triple")


def metadata(eng: Engine, cargo_features: str) -> dict:
    metadata_features: list[str] = []
    for feature in cargo_features.split(","):
        feature = feature.strip()
        if feature:
            metadata_features.extend(
                f"{crate}/{feature}" for crate in eng.metadata_feature_crates
            )

    args = [
        "cargo",
        "metadata",
        "--format-version=1",
        "--filter-platform",
        host_triple(eng.root),
    ]
    if metadata_features:
        args.extend(["--features", ",".join(metadata_features)])
    return json.loads(run_capture(args, cwd=eng.root))


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
        meta = metadata(eng, cargo_features)
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

        # Never drop a requested target crate, only its excluded dependencies.
        exclude = excluded_packages(eng, crates) - set(target_names)

        seen: set[str] = set()
        stack = [by_name[name]["id"] for name in target_names]
        closure = []
        while stack:
            package_id = stack.pop()
            if package_id in seen:
                continue
            seen.add(package_id)
            package = by_id[package_id]
            closure.append(package)

            for dep in resolve_nodes.get(package_id, {}).get("deps", []):
                dep_kinds = dep.get("dep_kinds", [])
                # An empty `dep_kinds` is a normal (non-dev) edge; only
                # drop deps whose every listed kind is `dev`.
                if dep_kinds and all(kind.get("kind") == "dev" for kind in dep_kinds):
                    continue
                dep_package = by_id.get(dep["pkg"])
                if dep_package is not None and dep_package.get("source") is None:
                    stack.append(dep_package["id"])

        for package in closure:
            if package["name"] in exclude:
                continue
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

    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *pathspecs],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    files = {
        Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw
    }
    return sorted(files, key=lambda path: path.as_posix())


def source_fingerprint(eng: Engine, crates: list[str], cargo_features: str) -> str:
    digest = hashlib.sha256()
    for path in fingerprint_inputs(eng, crates, cargo_features):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        full_path = eng.root / path
        if full_path.is_file():
            digest.update(full_path.read_bytes())
        else:
            # `git ls-files` includes tracked paths deleted in the working
            # tree. A deletion is part of the source state and must change the
            # fingerprint instead of making extraction unusable until commit.
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
) -> str:
    return "\n".join(
        [
            f"crate={crate}",
            f"platform={platform_key}",
            f"charon={charon_stamp}",
            f"features={cargo_features}",
            f"flags={' '.join(flags)}",
            f"charon_flags={' '.join(charon_flags)}",
            f"layout_targets={' '.join(layout_targets)}",
            f"source={source_fingerprint(eng, [crate], cargo_features)}",
        ]
    )


def crate_layout_targets(eng: Engine, spec: CrateSpec) -> list[str]:
    """Cross targets this crate emits a layout sidecar for.

    The extraction host is never in the list: its layouts already live in
    the crate's own artefact.
    """
    extra = eng.layout_targets if spec.layout_targets is None else spec.layout_targets
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
        stamp = stamp_for(
            eng,
            crate=crate,
            platform_key=platform_key,
            charon_stamp=charon_stamp,
            cargo_features=cargo_features,
            flags=flags,
            charon_flags=charon_flags,
            layout_targets=crate_layout_targets(eng, spec),
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
            layout_flags = (
                flags
                if spec.layout_cargo_args is None
                else [expand_features(arg, cargo_features) for arg in spec.layout_cargo_args]
            )
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


def run_cli(
    specs: dict[str, CrateSpec],
    default_crates: list[str],
    *,
    root: Path,
    out_dir: Path,
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
    extract(eng, args)
