#!/usr/bin/env python3
"""Stage Pyre's merged release stdlib, following PyPy's package.py layout."""

from __future__ import annotations

import argparse
import ast
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def ignored(_directory: str, names: list[str]) -> set[str]:
    ignored_names = {"__pycache__"} & set(names)
    ignored_names.update(name for name in names if name.endswith((".pyc", ".pyo", ".o", ".obj")))
    return ignored_names


def ensurepip_version(ensurepip: Path) -> str:
    """Read the bundled version without importing the stdlib being staged."""
    module = ast.parse(ensurepip.read_text(encoding="utf-8"), filename=str(ensurepip))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "_PIP_VERSION" for target in statement.targets):
            value = ast.literal_eval(statement.value)
            if isinstance(value, str):
                return value
    raise SystemExit(f"cannot determine _PIP_VERSION from {ensurepip}")


def verify_bundled_pip(destination: Path) -> None:
    version = ensurepip_version(destination / "ensurepip" / "__init__.py")
    wheel_dir = destination / "ensurepip" / "_bundled"
    expected = wheel_dir / f"pip-{version}-py3-none-any.whl"
    wheels = sorted(wheel_dir.glob("pip-*.whl"))
    if wheels != [expected]:
        names = ", ".join(path.name for path in wheels) or "<none>"
        raise SystemExit(
            f"ensurepip expects {expected.name}, staged pip wheels are: {names}"
        )
    try:
        with zipfile.ZipFile(expected) as wheel:
            corrupt = wheel.testzip()
            if corrupt is not None:
                raise SystemExit(f"corrupt member in {expected}: {corrupt}")
            if "pip/__init__.py" not in wheel.namelist():
                raise SystemExit(f"{expected} does not contain pip/__init__.py")
    except zipfile.BadZipFile as error:
        raise SystemExit(f"invalid bundled pip wheel {expected}: {error}") from error


def stage(destination: Path) -> None:
    destination = destination.resolve()
    allowed_root = (ROOT / "dist-assets").resolve()
    if destination == allowed_root or allowed_root not in destination.parents:
        raise SystemExit(f"destination must be below {allowed_root}: {destination}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # pypy/tool/release/package.py copies lib-python first, then overlays
    # lib_pypy into the same implementation-version directory.
    shutil.copytree(ROOT / "lib-python" / "3", destination, ignore=ignored)
    shutil.copytree(ROOT / "lib_pypy", destination, dirs_exist_ok=True, ignore=ignored)

    # `lib_pypy/__init__.py` only identifies the source-tree package.  In a
    # release its contents are the top level of the merged stdlib.
    init_file = destination / "__init__.py"
    if init_file.exists():
        init_file.unlink()

    if not (destination / "site.py").is_file():
        raise SystemExit("staged stdlib does not contain site.py")
    verify_bundled_pip(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    stage(args.destination)


if __name__ == "__main__":
    main()
