# Dummy file to make this directory a package.
#
# It stands in for CPython's own `test` package, which several parity scripts
# import for `test.support` helpers and which the standalone CPython builds
# this repo is commonly run against do not ship.  `__path__` points at the
# vendored copy, so the helper code is CPython's own.
#
# Only `test` is served this way.  Putting `lib-python/3` itself on the oracle's
# `PYTHONPATH` would shadow its entire stdlib with the vendored one, and an
# oracle running the tree under test answers nothing.
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[5] / "lib-python" / "3" / "test")]
