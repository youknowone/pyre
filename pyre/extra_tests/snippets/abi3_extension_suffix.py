# PyPy 5578: `_imp.extension_suffixes` advertises `.abi3.so` after the
# native suffix, and never a bare `.so`.  `site.py` installs the
# packaging.tags finder so pip will accept cp312+ abi3 wheels.
# CPython still lists a bare `.so`; this contract is PyPy/pyre only.
import sys

if sys.implementation.name not in ("pypy", "pyre"):
    raise SystemExit(0)
if sys.platform == "win32":
    raise SystemExit(0)

import _imp
import importlib.machinery

suffixes = _imp.extension_suffixes()
assert suffixes, suffixes
assert suffixes[0].endswith(".so"), suffixes
assert ".abi3.so" in suffixes, suffixes
assert ".so" not in suffixes, suffixes
assert ".abi3.so" in importlib.machinery.EXTENSION_SUFFIXES

import _pypy_abi3_tags

assert any(
    isinstance(finder, _pypy_abi3_tags._Abi3TagsFinder) for finder in sys.meta_path
)
