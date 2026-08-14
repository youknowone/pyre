#!/usr/bin/env python3
"""Stage Pyre's release stdlib.

This is the stdlib half of `pypy/tool/release/package.py:192-390 create_package`,
ported here rather than invoked: every function below has a counterpart there and
is cited by line.  Read that file before changing the layout.

The rest of `create_package` has no counterpart here.  `:233-240` runs the
`lib_pypy` cffi build scripts for `_ssl`, `sqlite3`, `_blake2`, `_sha3`,
`_tkinter` and the rest, which Pyre implements natively; `smartstrip` and
`make_portable` post-process a translated binary; and `:449-486` writes the zip
or tarball itself, which `dist` does for us out of the `include` entry in
`dist-workspace.toml`.  What is left of it would not run unmodified anyway:
`:221` spells the implementation directory `pypy{ver}`, so it would stage where
`sysconfig._get_implementation` does not look.

Two deliberate departures, both noted at the site: `lib_pypy` is not overlaid,
and the bundled pip wheel is verified against `ensurepip`.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# package.py:221 `IMPLEMENTATION = 'pypy{}'.format(python_ver)`.
IMPLEMENTATION = "pyre3.14"


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
    """No package.py counterpart.

    `ensurepip` derives the wheel it installs from `_PIP_VERSION`, so a wheel
    that does not match the name it builds fails when a user bootstraps pip
    rather than when the release is built.
    """
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


def stdlib_destination(assets_root: Path) -> Path:
    """package.py:222-227 — where the release keeps the stdlib on this platform.

    Windows holds the whole thing in `Lib`: that is where `sysconfig`'s `nt`
    scheme puts `stdlib` (`{installed_base}/Lib`) and where
    `site.getsitepackages` looks for `site-packages` (`<prefix>/Lib`).  Every
    other platform uses `<platlibdir>/<implementation><version>`, and
    `sys.platlibdir` is `lib`.
    """
    if os.name == "nt":
        return assets_root / "Lib"
    return assets_root / "lib" / IMPLEMENTATION


def stage(assets_root: Path) -> None:
    destination = stdlib_destination(assets_root).resolve()
    allowed_root = (ROOT / "dist-assets").resolve()
    if destination == allowed_root or allowed_root not in destination.parents:
        raise SystemExit(f"destination must be below {allowed_root}: {destination}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # pypy/tool/release/package.py copies lib-python first, then overlays
    # lib_pypy into the same implementation-version directory.  pyre stages
    # only `lib-python/3`: overlaying `lib_pypy` would put its cffi/pure-Python
    # shims (`_testcapi`, `_md5`, `_sha*`, `_sqlite3`, ...) on the release
    # import path, which is the same shadowing the source layout refuses.  What
    # pyre still owns from `lib_pypy` lives in `lib-python/3` instead.
    shutil.copytree(ROOT / "lib-python" / "3", destination, ignore=ignored)

    if not (destination / "site.py").is_file():
        raise SystemExit("staged stdlib does not contain site.py")
    verify_bundled_pip(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "assets_root",
        type=Path,
        help="directory whose contents dist copies into the archive root",
    )
    args = parser.parse_args()
    stage(args.assets_root)


if __name__ == "__main__":
    main()
