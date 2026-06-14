#!/usr/bin/env python3
"""Extract Charon ULLBC artefacts with source-fingerprint skip logic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ALL_CRATES = ["corpus", "pyre-object", "pyre-module", "pyre-interpreter", "pyre-jit"]
DEFAULT_CRATES = ["pyre-object", "pyre-interpreter", "pyre-jit"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def crate_info(root: Path, crate: str, cargo_features: str) -> tuple[Path, list[str]]:
    if crate == "corpus":
        return root / "majit" / "charon-corpus", []
    if crate == "pyre-object":
        return root / "pyre" / "pyre-object", []
    if crate == "pyre-module":
        return (
            root / "pyre" / "pyre-module",
            ["--features", f"pyre-interpreter/{cargo_features}"],
        )
    if crate == "pyre-interpreter":
        return root / "pyre" / "pyre-interpreter", ["--features", cargo_features]
    if crate == "pyre-jit":
        return root / "pyre" / "pyre-jit", ["--features", cargo_features]
    raise SystemExit(
        f"extract-llbc.py: unknown crate '{crate}'\n  known: {' '.join(ALL_CRATES)}"
    )


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


def metadata(root: Path, cargo_features: str) -> dict:
    metadata_features: list[str] = []
    for feature in cargo_features.split(","):
        feature = feature.strip()
        if feature:
            metadata_features.extend(
                [f"pyre-interpreter/{feature}", f"pyre-jit/{feature}"]
            )

    args = ["cargo", "metadata", "--format-version=1", "--filter-platform", host_triple(root)]
    if metadata_features:
        args.extend(["--features", ",".join(metadata_features)])
    return json.loads(run_capture(args, cwd=root))


def fingerprint_inputs(root: Path, crates: list[str], cargo_features: str) -> list[Path]:
    meta = metadata(root, cargo_features)
    packages = meta["packages"]
    by_name = {package["name"]: package for package in packages}
    by_id = {package["id"]: package for package in packages}
    resolve_nodes = {
        node["id"]: node for node in meta.get("resolve", {}).get("nodes", [])
    }

    target_names: list[str] = []
    pathspecs = [
        "Cargo.lock",
        "Cargo.toml",
        "scripts/extract-llbc.py",
        "scripts/install-charon.py",
    ]

    for crate in crates:
        if crate == "corpus":
            pathspecs.extend(
                ["majit/charon-corpus/Cargo.toml", "majit/charon-corpus/src/"]
            )
        else:
            target_names.append(crate)

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
        closure.append(package)

        for dep in resolve_nodes.get(package_id, {}).get("deps", []):
            if all(kind.get("kind") == "dev" for kind in dep.get("dep_kinds", [])):
                continue
            dep_package = by_id.get(dep["pkg"])
            if dep_package is not None and dep_package.get("source") is None:
                stack.append(dep_package["id"])

    for package in closure:
        package_dir = Path(package["manifest_path"]).resolve().parent
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


def source_fingerprint(root: Path, crates: list[str], cargo_features: str) -> str:
    digest = hashlib.sha256()
    for path in fingerprint_inputs(root, crates, cargo_features):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / path).read_bytes())
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


def charon_paths(root: Path) -> tuple[str, Path, Path]:
    platform_key, charon_exe = platform_info()
    repo_parent = root.parent
    shared = Path(os.environ.get("PYRE_SHARED_BUILD", repo_parent / ".pyre-build"))
    charon_dest = Path(
        os.environ.get("CHARON_DEST", shared / "charon" / platform_key)
    )
    return platform_key, charon_dest, charon_dest / charon_exe


def llbc_dest(root: Path) -> Path:
    dest = Path(os.environ.get("LLBC_DEST", root / "build" / "llbc"))
    if not dest.is_absolute():
        dest = root / dest
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def charon_version(charon_dest: Path) -> str:
    stamp = charon_dest / ".installed-version"
    return stamp.read_text().strip() if stamp.exists() else "unknown"


def stamp_for(
    *,
    root: Path,
    crate: str,
    platform_key: str,
    charon_stamp: str,
    cargo_features: str,
    flags: list[str],
) -> str:
    return "\n".join(
        [
            f"crate={crate}",
            f"platform={platform_key}",
            f"charon={charon_stamp}",
            f"features={cargo_features}",
            f"flags={' '.join(flags)}",
            f"source={source_fingerprint(root, [crate], cargo_features)}",
        ]
    )


def extract(args: argparse.Namespace) -> None:
    root = repo_root()
    cargo_features = os.environ.get("CARGO_FEATURES", "cranelift")
    platform_key, charon_dest, charon_bin = charon_paths(root)

    if not charon_bin.exists():
        raise SystemExit(
            f"extract-llbc.py: charon not installed at {charon_bin}\n"
            "  run: scripts/install-charon.py"
        )

    dest_dir = llbc_dest(root)
    charon_stamp = charon_version(charon_dest)
    env = os.environ.copy()
    prepend_msvc_link(env)

    crate_attr = "-Zcrate-attr=feature(cfg_select)"
    env["RUSTC_BOOTSTRAP"] = "1"
    env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " " + crate_attr).strip()
    env["CARGO_UNSTABLE_HOST_CONFIG"] = "true"
    env["CARGO_UNSTABLE_TARGET_APPLIES_TO_HOST"] = "true"
    host_config = [
        "--config",
        "target-applies-to-host=false",
        "--config",
        f'host.rustflags=["{crate_attr}"]',
    ]

    for crate in args.crates or ALL_CRATES:
        path, flags = crate_info(root, crate, cargo_features)
        if not path.is_dir():
            raise SystemExit(f"extract-llbc.py: missing crate dir for '{crate}' at {path}")

        dest = dest_dir / f"{crate}.ullbc"
        stamp_path = dest.with_suffix(dest.suffix + ".fingerprint")
        stamp = stamp_for(
            root=root,
            crate=crate,
            platform_key=platform_key,
            charon_stamp=charon_stamp,
            cargo_features=cargo_features,
            flags=flags,
        )

        if (
            not args.force
            and dest.exists()
            and dest.stat().st_size > 0
            and stamp_path.exists()
            and stamp_path.read_text() == stamp + "\n"
        ):
            print(f"=== skipping {crate} -> {dest} (fingerprint unchanged) ===")
            continue

        print(f"=== extracting {crate} -> {dest} ===")
        command = [
            str(charon_bin),
            "cargo",
            "--ullbc",
            "--dest-file",
            str(dest),
            "--",
            *flags,
            *host_config,
        ]
        subprocess.run(command, cwd=path, env=env, check=True)
        stamp_path.write_text(stamp + "\n")
        print(f"    wrote {dest} ({dest.stat().st_size} bytes)")

    print()
    print(f"all extractions complete. artefacts under: {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crates", nargs="*", help=f"known: {' '.join(ALL_CRATES)}")
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--list-inputs", action="store_true")
    parser.add_argument("--force", action="store_true", default=os.environ.get("LLBC_FORCE_REEXTRACT") == "1")
    args = parser.parse_args()

    root = repo_root()
    crates = args.crates or DEFAULT_CRATES
    cargo_features = os.environ.get("CARGO_FEATURES", "cranelift")
    if args.list_inputs:
        for path in fingerprint_inputs(root, crates, cargo_features):
            print(path.as_posix())
        return
    if args.fingerprint:
        print(source_fingerprint(root, crates, cargo_features))
        return
    extract(args)


if __name__ == "__main__":
    main()
