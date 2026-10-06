# CPython-suite gap: `test_exceptions` deletes `__cause__` / `__context__` /
# `__traceback__` and never `__suppress_context__`, whose storage is a
# `T_BOOL` member rather than a getset.
#
# parity-tests reason: `PyMember_SetOne` refuses to delete a numeric/char
# member with `can't delete numeric/char attribute`. `descr_delsuppresscontext`
# raises `__suppress_context__ may not be deleted`, the same wording the
# neighbouring getsets use for `__cause__`.
#
# pyre-check: pypy-diverges: `W_BaseException.descr_delsuppresscontext`
# reports `__suppress_context__ may not be deleted`.
e = ValueError()
try:
    del e.__suppress_context__
except TypeError as err:
    assert str(err) == "can't delete numeric/char attribute", err
else:
    raise AssertionError("expected TypeError")

try:
    del e.__cause__
except TypeError as err:
    assert str(err) == "__cause__ may not be deleted", err
else:
    raise AssertionError("expected TypeError")

print("OK")
