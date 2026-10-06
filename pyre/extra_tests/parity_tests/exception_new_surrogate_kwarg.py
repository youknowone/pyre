# CPython-suite gap: `test_exceptions` constructs through `Type(*args)` and
# never calls `BaseException.__new__(cls, *args, **kw)` with a keyword name
# that has no UTF-8 encoding.
#
# parity-tests reason: `descr_new_base_exception` does
# `args_w, kwds_w = __args__.unpack()` and `# ignore kwds`. `Arguments.unpack`
# keeps keyword names as `text_w` (`self._utf8`), so a lone surrogate is
# ignored rather than encoded. `W_OSError.descr_new` unpacks the same way
# and rejects a non-empty `kwds_w` with TypeError.
#
# pyre-check: pypy-diverges: `W_OSError.descr_new` says
# `OSError does not take keyword arguments`; `_PyArg_NoKeywords` says
# `OSError() takes no keyword arguments`.
LONE = chr(0xD800)

e = BaseException.__new__(ValueError, 1, **{LONE: 2})
assert type(e) is ValueError, type(e)
assert e.args == (1,), e.args

try:
    OSError(1, **{LONE: 2})
except TypeError as err:
    assert str(err) == "OSError() takes no keyword arguments", err
except UnicodeEncodeError as err:
    raise AssertionError("surrogate keyword name must not encode") from err
else:
    raise AssertionError("expected TypeError")

try:
    OSError.__new__(OSError, 1, **{LONE: 2})
except TypeError as err:
    assert str(err) == "OSError() takes no keyword arguments", err
except UnicodeEncodeError as err:
    raise AssertionError("surrogate keyword name must not encode") from err
else:
    raise AssertionError("expected TypeError")

print("OK")
