#!/usr/bin/env python3
"""In-process driver for one file of the vendored `extra_tests/` tree.

    <interpreter> pyre/extra_tests/upstream/driver.py <path/to/test_x.py>

The vendored tree at the repository root is PyPy's own suite and is written
against pytest: module-level `test_*` functions, `pytest.raises` as a context
manager, `pytest.skip` to opt out at runtime.  The `pytest.py` / `_pytest/`
copy that ships beside it predates Python 3.12 and imports `imp`, so it cannot
load under either interpreter here, and no pytest is installed.

Rather than copy the test bodies somewhere runnable, this driver supplies the
part of the pytest API those files actually use and executes them in place.
The shim is registered in `sys.modules` instead of being written as a
`pytest.py` file so that nothing else on `sys.path` is shadowed.

Output is one `PASS` / `SKIP` / `FAIL` line per test plus a summary line; the
exit code is 0 iff no test failed.  A file whose tests are all skipped still
exits 0 — the upstream files guard on `os.fork`, `/tmp`, `/bin/sh` and so on.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
import types
from pathlib import Path


class Skipped(Exception):
    """Raised by the shim's `skip()`; reported as SKIP, not as a failure."""


def _skip(reason: str = "") -> None:
    raise Skipped(reason)


class _RaisesContext:
    def __init__(self, expected):
        self.expected = expected
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected!r}")
        if issubclass(exc_type, self.expected):
            self.value = exc_value
            return True
        return False


def _raises(expected, *args, **kwargs):
    """`raises(Exc)` as a context manager, `raises(Exc, func, *a, **kw)` direct."""
    if not args:
        return _RaisesContext(expected)
    func, rest = args[0], args[1:]
    try:
        func(*rest, **kwargs)
    except expected as exc:
        return exc
    raise AssertionError(f"DID NOT RAISE {expected!r}")


def _install_pytest_shim() -> None:
    shim = types.ModuleType("pytest")
    shim.raises = _raises
    shim.skip = _skip
    shim.fail = lambda msg="": (_ for _ in ()).throw(AssertionError(msg))
    shim.Skipped = Skipped
    sys.modules["pytest"] = shim


def _load(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <test file>", file=sys.stderr)
        return 2
    path = Path(argv[1]).resolve()

    _install_pytest_shim()
    module = _load(path)

    # `vars()` preserves definition order, which is the order pytest collects in.
    tests = [
        (name, obj)
        for name, obj in vars(module).items()
        if name.startswith("test") and callable(obj)
    ]
    if not tests:
        print(f"{path.name}: no tests collected", file=sys.stderr)
        return 1

    passed = skipped = failed = 0
    for name, func in tests:
        try:
            func()
        except Skipped as exc:
            skipped += 1
            print(f"SKIP {name} ({exc})")
        except BaseException:  # noqa: BLE001 - report, don't abort the file
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {name}")

    print(f"{path.name}: {passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
