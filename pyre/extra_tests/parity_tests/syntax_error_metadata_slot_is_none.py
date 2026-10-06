# CPython-suite gap: `test_exceptions` never reads `SyntaxError._metadata`.
#
# parity-tests reason: `SyntaxError_init` writes `_metadata` as a
# `T_OBJECT` member, None when omitted. `W_SyntaxError` has no such
# slot, so pypy3 raises AttributeError.
#
# pyre-check: pypy-diverges: pypy3 has no `_metadata` attribute on
# SyntaxError.
e = SyntaxError("m", ("f.py", 1, 2, "line", 3, 4))
assert e._metadata is None, e._metadata
e2 = SyntaxError("m")
assert e2._metadata is None, e2._metadata

print("OK")
