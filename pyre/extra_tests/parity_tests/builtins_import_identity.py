"""Phase 5 Section 3 parity: `import builtins is space.builtin`.

PyPy's `import builtins` returns the singleton `space.builtin`
Module (`pypy/interpreter/module.py:18 Module.__init__` keeps one
Module per name; `Space.builtin` IS the builtins module).  Pyre
previously created a fresh `Module` on every
`load_builtin_module("builtins")` call, breaking identity against
`__builtins__` and EC's cached builtins module wrapper.

Fix: `load_module_dispatch` short-circuits "builtins" to
`ExecutionContext::get_builtin()` so the cached Module wrapper is
returned.
"""

import builtins
b = __builtins__
# __builtins__ at module level is the module itself (for the main
# script) or the dict (inside imported modules).  Either way the
# canonical module is reachable.
if hasattr(b, "__dict__"):
    assert b is builtins, (
        f"__builtins__ identity broken: {id(b):x} vs {id(builtins):x}"
    )
else:
    # b is the dict; the module that owns it is the same as `builtins`.
    assert b is builtins.__dict__, (
        f"__builtins__ dict identity broken: {id(b):x} vs {id(builtins.__dict__):x}"
    )

# Round-trip: builtin.__dict__ is the canonical builtins dict;
# mutations via either route are visible from the other.
sentinel_key = "__pq_builtins_identity"
try:
    builtins.__dict__[sentinel_key] = 42
    assert getattr(builtins, sentinel_key) == 42
finally:
    builtins.__dict__.pop(sentinel_key, None)

print("OK")
