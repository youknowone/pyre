# CPython-suite gap: `test_exceptions` constructs through `Type(*args)` and
# never calls `BaseException.__new__(cls, *args, **kw)`, which is the
# `descr_new_base_exception` path that unpacks then ignores `kwds_w`.
#
# parity-tests reason: `descr_new_base_exception` does
# `args_w, kwds_w = __args__.unpack()` and `# ignore kwds`, so
# `BaseException.__new__(ValueError, 1, foo=2)` stores `args == (1,)`.
# `W_OSError.descr_new` unpacks the same way and rejects a non-empty
# `kwds_w` when `_use_init` is false.
#
# pyre-check: pypy-diverges: `W_OSError.descr_new` says
# `OSError does not take keyword arguments`; `_PyArg_NoKeywords` says
# `OSError() takes no keyword arguments`.
e = BaseException.__new__(ValueError, 1, foo=2)
assert type(e) is ValueError, type(e)
assert e.args == (1,), e.args

try:
    OSError.__new__(OSError, 2, "x", filename="a")
except TypeError as err:
    assert str(err) == "OSError() takes no keyword arguments", err
else:
    raise AssertionError("expected TypeError")

print("OK")
