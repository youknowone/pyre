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
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# package.py:221 `IMPLEMENTATION = 'pypy{}'.format(python_ver)`.  The trailing
# `t` is the free-threaded ABI flag: `sysconfig` derives `abi_thread` from
# `Py_GIL_DISABLED` and the posix schemes put it in the directory name.
IMPLEMENTATION = "pyre3.14t"


def extension_suffix() -> str:
    """The suffix `cpyext::extension_suffix` publishes on supported hosts."""
    if sys.platform == "darwin":
        return ".pyre314-darwin.so"
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return ".pyre314-x86_64-linux-gnu.so"
    if machine in {"aarch64", "arm64"}:
        return ".pyre314-aarch64-linux-gnu.so"
    return ".pyre314-linux-gnu.so"


def build_sqlite3_cffi(destination: Path) -> None:
    """PyPy `package.py:create_package`'s `_sqlite3_build.py` entry.

    The generated library must be compiled by PyPy so cffi selects its
    `_cffi_pypyinit_*` form.  Pyre's extension loader consumes that immutable
    type context directly; renaming only the import suffix does not alter the
    native ABI.
    """
    if os.name == "nt":
        # Pyre's cpyext/host-dlopen feature is currently enabled only on macOS
        # and Linux, so publishing an un-loadable Windows extension would hide
        # the clean public-module fallback failure.
        return
    pypy = os.environ.get("PYPY3") or shutil.which("pypy3")
    if not pypy:
        raise SystemExit("staging sqlite3 requires pypy3 (or PYPY3)")
    helper = r"""
import ctypes.util
import sys

source_root, output_root = sys.argv[1:]
if sys.platform == "darwin":
    original_find_library = ctypes.util.find_library
    ctypes.util.find_library = lambda name: (
        "/usr/lib/libsqlite3.dylib"
        if name == "sqlite3" and original_find_library(name) is None
        else original_find_library(name)
    )
sys.path.insert(0, source_root)
import _sqlite3_build
print("PYRE_SQLITE3_CFFI=" + _sqlite3_build._ffi.compile(
    tmpdir=output_root,
    verbose=True,
))
"""
    with tempfile.TemporaryDirectory(prefix="pyre-sqlite3-cffi-") as temporary:
        result = subprocess.run(
            [pypy, "-c", helper, str(ROOT / "lib_pypy"), temporary],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        marker = "PYRE_SQLITE3_CFFI="
        outputs = [line.removeprefix(marker) for line in result.stdout.splitlines() if line.startswith(marker)]
        if len(outputs) != 1:
            raise SystemExit("_sqlite3_build.py did not report exactly one extension path")
        shutil.copy2(outputs[0], destination / f"_sqlite3_cffi{extension_suffix()}")
    shutil.copy2(ROOT / "lib_pypy" / "_sqlite3.py", destination / "_sqlite3.py")


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
    # lib_pypy into the same implementation-version directory.  Pyre selects
    # only the app-level owners it still needs: overlaying all of lib_pypy
    # would expose CPython-only test shims such as `_testcapi` and suppress
    # intended fallbacks for modules pyre owns natively.
    shutil.copytree(ROOT / "lib-python" / "3", destination, ignore=ignored)
    build_sqlite3_cffi(destination)

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
