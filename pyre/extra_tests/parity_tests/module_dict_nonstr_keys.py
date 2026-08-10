# CPython-suite gap: module tests omit non-string keys on PyPy ModuleDictStrategy.
# parity-tests reason: this targets PyPy/pyre module-dict strategy conversion.

"""Phase 5+ parity test: non-str keys on W_ModuleDictObject.

PyPy `pypy/objspace/std/celldict.py:67-74 setitem` switches to the
object strategy when a non-str key arrives; subsequent reads /
writes / deletes hit `ObjectDictStrategy` (`dictmultiobject.py:21
_never_equal_to_string`).  Pyre's port mirrors this via
`w_module_dict_switch_to_object_strategy` (drains the str entries
into a unified `object_storage: Vec<(PyObjectRef, PyObjectRef)>`
and bumps `mstrategy.version`); after the switch every key type
goes through the same Vec via `dict_keys_equal`.

Targets `__builtins__` (the canonical W_ModuleDictObject in pyre)
to exercise the dispatch entry points that route through
W_ModuleDictObject's str / object-strategy fork.

Cleanup is best-effort because mutations on `__builtins__` would
otherwise leak into subsequent tests.
"""

# `__builtins__` can be either a module wrapper or the dict itself
# depending on how the script was loaded; reach the dict in both
# shapes.
b = __builtins__
bd = b.__dict__ if hasattr(b, "__dict__") else b
assert isinstance(bd, dict), f"type(__builtins__): {type(b).__name__}"

# Use a sentinel int key well outside any plausible builtin id range
_K_INT = -99887766
_K_NONE = None
try:
    # (1) Store + read a non-str (int) key.
    bd[_K_INT] = "one"
    assert bd[_K_INT] == "one", f"int-key read: {bd[_K_INT]!r}"
    assert _K_INT in bd, "int key must be present after store"

    # (2) Overwrite an existing non-str key.
    bd[_K_INT] = "ONE"
    assert bd[_K_INT] == "ONE", f"int-key overwrite: {bd[_K_INT]!r}"

    # (3) Store None-key — PyPy's `_never_equal_to_string` family.
    bd[_K_NONE] = "n"
    assert bd[_K_NONE] == "n", f"None-key read: {bd[_K_NONE]!r}"

    # (4) `dict.pop()` on a non-str key removes the entry without the
    #     prior W_DictObject mis-cast on W_ModuleDictObject.
    popped = bd.pop(_K_INT)
    assert popped == "ONE"
    assert _K_INT not in bd, "after pop, int key must be absent"

    # (5) `del` on a non-str key.
    del bd[_K_NONE]
    assert _K_NONE not in bd
finally:
    # Defensive cleanup if any assertion above raised.
    bd.pop(_K_INT, None)
    bd.pop(_K_NONE, None)

print("OK")
