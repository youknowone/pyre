# CPython-suite gap: `test_exceptions` stringifies a SyntaxError with a
# filename and lineno, never with `end_lineno` larger than `lineno`.
#
# parity-tests reason: `SyntaxError_str` prints `line N` from `lineno`
# alone. `W_SyntaxError.descr_str` prints `lines N-M` when `end_lineno`
# is larger. That method has no `@jit` hint.
#
# pyre-check: pypy-diverges: pypy3 answers `m (f.py, lines 1-3)` for a
# six-field details tuple whose `end_lineno` is 3.
e = SyntaxError("m", ("f.py", 1, 2, "line", 3, 4))
assert str(e) == "m (f.py, line 1)", str(e)
e.msg = 5
assert str(e) == "5 (f.py, line 1)", str(e)

print("OK")
