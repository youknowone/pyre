# CPython-suite gap: `test_import` exercises `from pkg import sub` and star
# imports through the app-level importer only. Nothing there pins the contract
# itself -- which names get bound, which failures are swallowed, and what a
# cached submodule does -- so a runtime that has to implement `_handle_fromlist`
# somewhere other than `importlib._bootstrap` is graded by nothing.
# parity-tests reason: pyre answers `__import__` from two importers. The
# app-level `_bootstrap.__import__` runs wherever `importlib._bootstrap` is in
# `sys.modules`; the native importer stands in while it is not -- during
# startup, and for the whole run of a guest that carries no importlib, where
# `from . import X` reaches nothing else. Both owe the same observable
# behaviour, and only the app-level one is covered by the suite.

import json
import os
import shutil
import sys
import tempfile

root = tempfile.mkdtemp(prefix="pyre_hfl_")
os.mkdir(os.path.join(root, "hflpkg"))
for name, body in [
    ("__init__.py", "__all__ = ['a', 'b']\n"),
    ("a.py", "A = 1\n"),
    ("b.py", "B = 2\n"),
    ("c.py", "C = 3\n"),
]:
    with open(os.path.join(root, "hflpkg", name), "w") as f:
        f.write(body)
sys.path.insert(0, root)

# `*` expands `__all__` and imports each name in it; a submodule the package
# does not list stays unimported.
mod = __import__("hflpkg", {}, {}, ["*"], 0)
assert mod.__name__ == "hflpkg"
assert hasattr(mod, "a") and hasattr(mod, "b"), "star did not import __all__"
assert not hasattr(mod, "c"), "star imported a name outside __all__"
assert "hflpkg.c" not in sys.modules

# A fromlist name that is neither an attribute nor an importable submodule is
# swallowed: `__import__` returns the package and IMPORT_FROM raises later.
assert __import__("hflpkg", {}, {}, ["zzz"], 0).__name__ == "hflpkg"

# A submodule whose own body raises ModuleNotFoundError for a *different* name
# is a real failure, not a name the fromlist may skip.
with open(os.path.join(root, "hflpkg", "d.py"), "w") as f:
    f.write("import nonexistent_module_xyz\n")
try:
    __import__("hflpkg", {}, {}, ["d"], 0)
except ModuleNotFoundError as exc:
    assert exc.name == "nonexistent_module_xyz", exc.name
else:
    raise AssertionError("expected ModuleNotFoundError")

# `sys.modules[name] = None` blocks a name; the fromlist does not swallow that
# either.
sys.modules["hflpkg.blocked"] = None
try:
    __import__("hflpkg", {}, {}, ["blocked"], 0)
except ImportError as exc:
    assert "hflpkg.blocked" in str(exc), str(exc)
else:
    raise AssertionError("expected ImportError for a blocked name")
del sys.modules["hflpkg.blocked"]

# Every item must be a str, and the message names where the item came from.
try:
    __import__("hflpkg", {}, {}, [1], 0)
except TypeError as exc:
    assert str(exc) == "Item in ``from list'' must be str, not int", str(exc)
else:
    raise AssertionError("expected TypeError")

mod.__all__ = ["a", 2]
try:
    __import__("hflpkg", {}, {}, ["*"], 0)
except TypeError as exc:
    assert str(exc) == "Item in hflpkg.__all__ must be str, not int", str(exc)
else:
    raise AssertionError("expected TypeError")
mod.__all__ = ["a", "b"]

# A `sys.modules` hit returns before the parent binding runs, so neither
# spelling puts back an attribute the program deleted.
import json.decoder

decoder = sys.modules["json.decoder"]
delattr(json, "decoder")
import json.decoder

assert not hasattr(json, "decoder"), "`import a.b` rebound a cached submodule"

got = __import__("json", {}, {}, ["decoder"], 0)
assert not hasattr(got, "decoder"), "a fromlist rebound a cached submodule"
assert sys.modules["json.decoder"] is decoder, "the cached submodule was reloaded"

# A fromlist item is an arbitrary `str`, including one carrying a lone
# surrogate. Such a name cannot be an importable module, so the optional-import
# failure is swallowed and the package comes back.
assert __import__("hflpkg", {}, {}, ["\ud800"], 0) is mod
setattr(mod, "\ud800", 5)
assert __import__("hflpkg", {}, {}, ["\ud800"], 0) is mod
delattr(mod, "\ud800")

# `sys.modules` may hold any object with a `__path__`. The child name is
# formatted from `__name__`, so a non-str one is coerced rather than rejected,
# and the resulting import failure names the parent -- not the child -- so it
# is not the one a fromlist swallows.
class Package:
    __path__ = []
    __name__ = 123


sys.modules["hflfake"] = Package()
try:
    __import__("hflfake", {}, {}, ["x"], 0)
except ModuleNotFoundError as exc:
    assert exc.name == "123", exc.name
else:
    raise AssertionError("expected ModuleNotFoundError")
del sys.modules["hflfake"]

# The fromlist is iterated lazily. An item is fully processed before the next is
# requested, so an error a later `__next__` raises cannot preempt the TypeError
# an earlier item already owes...
class NonStrThenRaise:
    def __init__(self):
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        if self.n == 1:
            return 1
        raise ValueError("must not be reached before the TypeError")


try:
    __import__("hflpkg", {}, {}, NonStrThenRaise(), 0)
except TypeError as exc:
    assert str(exc) == "Item in ``from list'' must be str, not int", str(exc)
else:
    raise AssertionError("expected TypeError")

# ... and an import an earlier item performed is visible to the `__next__` that
# follows it.
class WatchAfterFirst:
    def __init__(self):
        self.n = 0
        self.saw = None

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        if self.n == 1:
            return "c"
        self.saw = hasattr(sys.modules["hflpkg"], "c")
        raise StopIteration


watcher = WatchAfterFirst()
__import__("hflpkg", {}, {}, watcher, 0)
assert watcher.saw is True, "the earlier item's import was not visible"

sys.path.remove(root)
shutil.rmtree(root, ignore_errors=True)
print("OK")
