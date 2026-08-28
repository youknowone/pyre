# CPython-suite gap: module tests omit mixed-key order across ModuleDictStrategy exit.
# parity-tests reason: this targets PyPy/pyre module-dict storage transitions.

"""Phase 5 parity: mixed-key insertion order on W_ModuleDictObject.

`celldict.py:173-186 switch_to_object_strategy` drains existing str
entries into the new ObjectDictStrategy storage in their original
insertion order, then re-dispatches the triggering setitem.  All
subsequent setitems append to the same unified Vec, so
`["a", 1, "b"]` insertion order is preserved end-to-end.

popitem() LIFO parity follows from the unified order: the most
recently inserted entry (regardless of key type) pops first.

Exercised against `__builtins__` since pyre's running-frame
`globals()` returns a W_DictObject today.
"""

b = __builtins__
bd = b.__dict__ if hasattr(b, "__dict__") else b

# Stash + restore: every mutation reverted via popitem-symmetric
# cleanup at the end.
_KS = ("__pq_mixed_a", -42, "__pq_mixed_b")
try:
    bd[_KS[0]] = 1
    bd[_KS[1]] = 2
    bd[_KS[2]] = 3

    # Mixed-key insertion order preserved through items().
    items = list(bd.items())
    # The three keys we just inserted must appear in insertion order
    # somewhere in the items list — they should be the *last* three
    # because we inserted them last.
    suffix = items[-3:]
    assert suffix == [(_KS[0], 1), (_KS[1], 2), (_KS[2], 3)], (
        f"mixed-key order broken: {suffix!r}"
    )

    # popitem LIFO: the most recently inserted entry pops first, regardless
    # of its key type.  Pop all three and verify reverse order.
    p1 = bd.popitem()
    assert p1 == (_KS[2], 3), f"popitem 1: {p1!r}"
    p2 = bd.popitem()
    assert p2 == (_KS[1], 2), f"popitem 2: {p2!r}"
    p3 = bd.popitem()
    assert p3 == (_KS[0], 1), f"popitem 3: {p3!r}"
finally:
    # Defensive: if any assertion raised, scrub leftover keys.
    for k in _KS:
        bd.pop(k, None)

print("OK")
