# CPython-suite gap: `test_exception_group` always constructs with two
# positionals and never `ExceptionGroup("m")` or a third positional.
#
# parity-tests reason: `BaseExceptionGroup_new` is a two-positional
# Clinic wrapper, so one or three arguments report
# `takes exactly 2 arguments (N given)`. `W_BaseExceptionGroup.descr_new`
# is interp2app with `w_message` and `w_exceptions`.
#
# pyre-check: pypy-diverges: one argument is
# `BaseExceptionGroup.__new__() missing 1 required positional argument:
# 'exceptions'`. Three arguments are
# `BaseExceptionGroup.__new__() takes 3 positional arguments but 4 were
# given`.
try:
    ExceptionGroup("m")
except TypeError as err:
    assert str(err) == (
        "BaseExceptionGroup.__new__() takes exactly 2 arguments (1 given)"
    ), err
else:
    raise AssertionError("expected TypeError")

try:
    ExceptionGroup("m", [ValueError()], 1)
except TypeError as err:
    assert str(err) == (
        "BaseExceptionGroup.__new__() takes exactly 2 arguments (3 given)"
    ), err
else:
    raise AssertionError("expected TypeError")

print("OK")
