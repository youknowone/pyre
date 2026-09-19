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
import re
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path


class Skipped(Exception):
    """Raised by the shim's `skip()`; reported as SKIP, not as a failure."""


def _skip(reason: str = "") -> None:
    raise Skipped(reason)


class _RaisesContext:
    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.value = None
        self.type = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected!r}")
        if issubclass(exc_type, self.expected):
            if self.match is not None and re.search(self.match, str(exc_value)) is None:
                raise AssertionError(
                    f"{self.match!r} does not match {str(exc_value)!r}"
                )
            self.value = exc_value
            self.type = exc_type
            return True
        return False


class _ExceptionInfo:
    """The small `py.test.raises` result surface PyPy's app-level tests use."""

    def __init__(self, value):
        self.value = value
        self.type = type(value)


def _raises(expected, *args, **kwargs):
    """`raises(Exc)` as a context manager, `raises(Exc, func, *a, **kw)` direct."""
    match = kwargs.pop("match", None)
    if not args:
        return _RaisesContext(expected, match)
    func, rest = args[0], args[1:]
    try:
        func(*rest, **kwargs)
    except expected as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(f"{match!r} does not match {str(exc)!r}")
        return _ExceptionInfo(exc)
    raise AssertionError(f"DID NOT RAISE {expected!r}")


class _Mark:
    def skipif(self, condition, *, reason=""):
        def decorate(func):
            if not condition:
                return func

            def skipped(*args, **kwargs):
                _skip(reason)

            skipped.__name__ = func.__name__
            return skipped

        return decorate


def _importorskip(name, minversion=None):
    try:
        return __import__(name)
    except ImportError as exc:
        _skip("could not import %r: %s" % (name, exc))


def _fixture(fn=None, **_kwargs):
    """`@pytest.fixture` / `@pytest.fixture()` — mark a factory for injection."""

    def decorate(func):
        func._pytest_fixture = True
        return func

    if callable(fn):
        return decorate(fn)
    return decorate


class _MonkeyPatch:
    """The `syspath_prepend` / `delitem` slice `test_abi3_tags.py` uses."""

    def __init__(self):
        self._undos = []

    def syspath_prepend(self, path):
        path = str(path)
        sys.path.insert(0, path)

        def undo():
            try:
                sys.path.remove(path)
            except ValueError:
                pass

        self._undos.append(undo)

    def delitem(self, mapping, key, raising=True):
        if key not in mapping:
            if raising:
                raise KeyError(key)
            return
        old = mapping[key]
        del mapping[key]
        self._undos.append(lambda m=mapping, k=key, v=old: m.__setitem__(k, v))

    def undo(self):
        while self._undos:
            self._undos.pop()()


def _params(fn):
    code = fn.__code__
    return code.co_varnames[: code.co_argcount]


def _resolve(name, fixtures, cache, undos):
    if name in cache:
        return cache[name]
    if name == "tmp_path":
        tmp = Path(tempfile.mkdtemp(prefix="pyre-upstream-"))
        undos.append(lambda: shutil.rmtree(tmp, ignore_errors=True))
        cache[name] = tmp
        return tmp
    if name == "monkeypatch":
        patch = _MonkeyPatch()
        undos.append(patch.undo)
        cache[name] = patch
        return patch
    if name not in fixtures:
        raise TypeError("unknown fixture %r" % (name,))
    kwargs = {
        param: _resolve(param, fixtures, cache, undos) for param in _params(fixtures[name])
    }
    value = fixtures[name](**kwargs)
    cache[name] = value
    return value


def _install_pytest_shim() -> None:
    shim = types.ModuleType("pytest")
    shim.raises = _raises
    shim.skip = _skip
    shim.fail = lambda msg="": (_ for _ in ()).throw(AssertionError(msg))
    shim.Skipped = Skipped
    shim.mark = _Mark()
    shim.importorskip = _importorskip
    shim.fixture = _fixture
    sys.modules["pytest"] = shim


def _load(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    # PyPy's app-level test collector publishes these helpers as globals;
    # `pypy/objspace/std/test/apptest_*.py` therefore deliberately does not
    # import them.  Keep the upstream source untouched and reproduce that
    # collector contract here.
    module.raises = _raises
    module.skip = _skip
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <test file>", file=sys.stderr)
        return 2
    path = Path(argv[1]).resolve()

    _install_pytest_shim()
    try:
        module = _load(path)
    except Skipped as exc:
        print(f"SKIP {path.name} ({exc})")
        return 0

    fixtures = {
        name: obj
        for name, obj in vars(module).items()
        if callable(obj) and getattr(obj, "_pytest_fixture", False)
    }
    # `vars()` preserves definition order, which is the order pytest collects in.
    tests = [
        (name, obj)
        for name, obj in vars(module).items()
        if name.startswith("test")
        and callable(obj)
        and not getattr(obj, "_pytest_fixture", False)
    ]
    if not tests:
        print(f"{path.name}: no tests collected", file=sys.stderr)
        return 1

    passed = skipped = failed = 0
    for name, func in tests:
        cache = {}
        undos = []
        try:
            kwargs = {
                param: _resolve(param, fixtures, cache, undos) for param in _params(func)
            }
            func(**kwargs)
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
        finally:
            for undo in reversed(undos):
                undo()

    print(f"{path.name}: {passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
