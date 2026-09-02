# CPython-suite gap: nothing in `test_import` reaches an importer other than
# `importlib._bootstrap`, so a runtime that carries a second implementation of
# `_handle_fromlist` is graded on only one of them.
# parity-tests reason: pyre's `dunder_import` hands a fromlist to the
# app-level `_bootstrap.__import__` whenever `importlib._bootstrap` is in
# `sys.modules`, and to a native `handle_fromlist` when it is not -- during
# startup, and for the whole run of a guest that carries no importlib. Only
# the first is reachable from an ordinary script, and `handle_fromlist_contract`
# therefore grades only the first. Dropping the `sys.modules` entry moves pyre
# onto the native importer while leaving CPython and PyPy untouched: both bind
# `__import__` to the frozen bootstrap at startup and never consult that key
# again, so every expected value below is the one the reference already gives.

import os
import shutil
import sys
import tempfile

root = tempfile.mkdtemp(prefix="pyre_hfn_")
os.mkdir(os.path.join(root, "hfnpkg"))
for name, body in [
    ("__init__.py", "__all__ = ['a', 'b']\n"),
    ("a.py", "A = 1\n"),
    ("b.py", "B = 2\n"),
    ("c.py", "C = 3\n"),
]:
    with open(os.path.join(root, "hfnpkg", name), "w") as f:
        f.write(body)
sys.path.insert(0, root)

# Everything the rows below need is imported before the switch, so the native
# importer is asked only for `hfnpkg`.
saved_bootstrap = sys.modules.pop("importlib._bootstrap", None)


def drop():
    for key in [k for k in sys.modules if k == "hfnpkg" or k.startswith("hfnpkg.")]:
        del sys.modules[key]


# `*` expands `__all__`; a submodule outside it stays unimported.
drop()
pkg = __import__("hfnpkg", {}, {}, ["*"], 0)
assert hasattr(pkg, "a") and hasattr(pkg, "b"), "star did not import __all__"
assert not hasattr(pkg, "c"), "star imported a name outside __all__"

# `elif x == '*'` is a comparison the item answers: a `str` subclass whose
# `__eq__` denies `'*'` is an ordinary name, whatever its text reads.
class NotStar(str):
    def __eq__(self, other):
        return False

    def __hash__(self):
        return str.__hash__(self)


drop()
pkg = __import__("hfnpkg", {}, {}, [NotStar("*")], 0)
assert not hasattr(pkg, "a"), "a denying __eq__ still expanded __all__"

# `from_name = f'{module.__name__}.{x}'` formats the item, so a `__format__`
# override names the module that gets imported ...
class FormatsToB(str):
    def __format__(self, spec):
        return "b"


drop()
__import__("hfnpkg", {}, {}, [FormatsToB("zzz")], 0)
assert "hfnpkg.b" in sys.modules, "the item's __format__ did not build the name"

# ... and one that raises propagates instead of being swallowed the way a
# missing submodule is.
class FormatRaises(str):
    def __format__(self, spec):
        raise ValueError("boom")


drop()
try:
    __import__("hfnpkg", {}, {}, [FormatRaises("zzz")], 0)
except ValueError as exc:
    assert str(exc) == "boom", str(exc)
else:
    raise AssertionError("expected ValueError")

# The package's own `__name__` is formatted by the same f-string.
class FormatsToPkg(str):
    def __format__(self, spec):
        return "hfnpkg"


drop()
pkg = __import__("hfnpkg", {}, {}, [], 0)
pkg.__name__ = FormatsToPkg("wrongname")
__import__("hfnpkg", {}, {}, ["b"], 0)
assert "hfnpkg.b" in sys.modules, "the __name__ override was not formatted"

# `if not fromlist` is one truth test wherever the call lands.
class CountingList(list):
    calls = 0

    def __bool__(self):
        type(self).calls += 1
        return list.__len__(self) != 0


drop()
__import__("hfnpkg", {}, {}, CountingList(["a"]), 0)
assert CountingList.calls == 1, CountingList.calls
CountingList.calls = 0
__import__("hfnpkg", {}, {}, CountingList(["a"]), 0)
assert CountingList.calls == 1, CountingList.calls

# A name that is neither an attribute nor an importable submodule is swallowed;
# an explicit `None` block is not.
drop()
assert __import__("hfnpkg", {}, {}, ["zzz"], 0).__name__ == "hfnpkg"
sys.modules["hfnpkg.blocked"] = None
try:
    __import__("hfnpkg", {}, {}, ["blocked"], 0)
except ImportError as exc:
    assert "hfnpkg.blocked" in str(exc), str(exc)
else:
    raise AssertionError("expected ImportError")
del sys.modules["hfnpkg.blocked"]

# A non-str item names where it was found.
drop()
try:
    __import__("hfnpkg", {}, {}, [1], 0)
except TypeError as exc:
    assert str(exc) == "Item in ``from list'' must be str, not int", str(exc)
else:
    raise AssertionError("expected TypeError")

if saved_bootstrap is not None:
    sys.modules["importlib._bootstrap"] = saved_bootstrap
sys.path.remove(root)
shutil.rmtree(root, ignore_errors=True)
print("OK")
