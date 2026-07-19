"""Phase 5+ parity test: runtime `__builtins__` mutations propagate.

PyPy `pypy/interpreter/baseobjspace.py:642 ObjSpace.builtin` is the
single `space.builtin.w_dict` consulted by `pick_builtin` and the
LOAD_GLOBAL builtin fallback (`pyopcode.py:558-565`).  Any mutation
on that dict is observable through every frame's LOAD_GLOBAL builtin
fallback.  Pyre likewise keeps the live builtins module as the frame's
builtin owner; builtin names are not copied into module globals.
"""

# A module namespace owns only its definitions plus `__builtins__`; names
# supplied by the builtin fallback must not leak into globals()/dir(module).
assert "__builtins__" in globals()
assert "len" not in globals()
assert "Ellipsis" not in globals()
assert len(()) == 0

bd = __builtins__.__dict__ if hasattr(__builtins__, "__dict__") else __builtins__

# (1) Inject a new builtin via `__builtins__.foo = bar`.
bd["_runtime_added_builtin"] = 42

# (2) Defining a new function should see the freshly added builtin
#     through its live builtins fallback.
def f():
    return _runtime_added_builtin  # noqa: F821 -- injected at runtime

assert f() == 42, f"expected 42, got {f()!r}"

# (3) Mutating the existing entry should also propagate.
bd["_runtime_added_builtin"] = 99
assert f() == 99, f"expected 99, got {f()!r}"

# (4) Cleanup.
del bd["_runtime_added_builtin"]

print("OK")
