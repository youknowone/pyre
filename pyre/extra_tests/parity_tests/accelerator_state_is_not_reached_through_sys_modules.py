# pyre-check: pypy-diverges: pypy3's `json` reaches `_json` by name, so every
# row below that blocks it raises `KeyError`, and its `csv.reader` reads the
# blocked entry as well.
#
# CPython-suite gap: `test_csv` and `test_json` exercise these accelerators
# thoroughly and `test_importlib` blocks names with the `None` sentinel, but no
# test does both at once, so nothing notices an accelerator reaching its own
# module by name.
#
# parity-tests reason: `_csvstate` and the encoder's `fast_encode` are module
# state, reached off the defining class and through a function pointer -- and a
# function pointer is the same one after a re-import, where an attribute is a
# fresh object.  Neither asks `sys.modules` where the module is, so
# `sys.modules[name] = None`, the documented way a program stops everything
# downstream from importing that name, does not reach them.  An accelerator
# that looks itself up by name instead fails on a module the program never
# asked it to use.
import io
import json
import sys

import csv


def blocked(name, fn):
    saved = sys.modules[name]
    sys.modules[name] = None
    try:
        return fn()
    finally:
        sys.modules[name] = saved


csv.register_dialect("parity", delimiter=";")
try:
    # The registry is state, so every operation over it answers while the name
    # is blocked -- reads, a write, and the refusal for a name never registered.
    assert blocked("_csv", lambda: csv.get_dialect("parity").delimiter) == ";"
    assert "parity" in blocked("_csv", csv.list_dialects)
    assert blocked("_csv", lambda: list(csv.reader(io.StringIO("a;b\n"), "parity"))) == [
        ["a", "b"]
    ]
    blocked("_csv", lambda: csv.register_dialect("parity2", delimiter="|"))
    assert "parity2" in csv.list_dialects()
    try:
        blocked("_csv", lambda: csv.unregister_dialect("never-registered"))
    except csv.Error as exc:
        assert str(exc) == "unknown dialect", exc
    else:
        raise AssertionError("unregister accepted an unknown dialect")
finally:
    csv.unregister_dialect("parity")
    csv.unregister_dialect("parity2")

assert "parity" not in csv.list_dialects(), csv.list_dialects()

# The encoder recognizes its own escaper by pointer, and both escapers are
# reachable that way, so neither `ensure_ascii` arm owes anything to the name.
assert blocked("_json", lambda: json.dumps({"a": 1, "b": [1, 2]})) == '{"a": 1, "b": [1, 2]}'
assert blocked("_json", lambda: json.dumps({"\xe9": 1}, ensure_ascii=False)) == '{"\xe9": 1}'
assert blocked("_json", lambda: json.dumps({"\xe9": 1})) == '{"\\u00e9": 1}'
assert blocked("_json", lambda: json.dumps({"b": 1, "a": 2}, sort_keys=True)) == '{"a": 2, "b": 1}'
assert blocked("_json", lambda: json.loads('{"a": 1}')) == {"a": 1}

# A re-import mints fresh escapers, and an encoder built against the previous
# ones is still the same escaper, so the fast path is not lost with the module.
first = json.JSONEncoder()
del sys.modules["_json"]
import _json  # noqa: F401

assert "".join(first.iterencode({"a": 1})) == '{"a": 1}'
assert json.dumps({"a": 1}) == '{"a": 1}'

print("OK")
