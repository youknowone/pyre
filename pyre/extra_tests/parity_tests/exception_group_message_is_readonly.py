# CPython-suite gap: `test_exception_group` reads `message` and
# `exceptions` and never assigns either field.
#
# parity-tests reason: `BaseExceptionGroup` installs `message` and
# `exceptions` as readonly members, so a store reports
# `readonly attribute`. `interp_exceptions.py` uses
# `GetSetProperty(..., fset=None)` and names the field:
# `readonly attribute 'message'`.
#
# pyre-check: pypy-diverges: pypy3 answers
# `readonly attribute 'message'` / `readonly attribute 'exceptions'`.
eg = ExceptionGroup("m", [ValueError(1)])
try:
    eg.message = "x"
except AttributeError as err:
    assert str(err) == "readonly attribute", err
else:
    raise AssertionError("expected AttributeError")

try:
    eg.exceptions = (ValueError(1),)
except AttributeError as err:
    assert str(err) == "readonly attribute", err
else:
    raise AssertionError("expected AttributeError")

print("OK")
