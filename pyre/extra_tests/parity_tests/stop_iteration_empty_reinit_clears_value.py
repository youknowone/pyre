# CPython-suite gap: `test_exceptions` reads `StopIteration.value` on a
# freshly constructed instance and never calls `__init__` again with no
# arguments, so an empty re-init that left the old payload would still pass.
#
# parity-tests reason: `StopIteration_init` always `Py_CLEAR`s `value` and
# then stores the first positional argument or `None`. `W_StopIteration.descr_init`
# writes `w_value` only when `args_w` is non-empty, so pypy3 keeps the old
# payload after `e.__init__()`.
#
# pyre-check: pypy-diverges: `W_StopIteration.descr_init` leaves `w_value`
# alone when `args_w` is empty, so pypy3 answers `StopIteration(1)` then
# `e.__init__()` with `e.value is 1`.
e = StopIteration(1)
e.__init__()
assert e.value is None, e.value
assert e.args == (), e.args

e = StopIteration(1)
e.__init__(2)
assert e.value == 2, e.value
assert e.args == (2,), e.args

print("OK")
