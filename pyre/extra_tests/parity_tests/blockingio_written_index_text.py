# CPython-suite gap: `test_exceptions` assigns an int to
# `BlockingIOError.characters_written` and never a str.
#
# parity-tests reason: `OSError_written_set` converts through
# `PyNumber_AsSsize_t`, whose missing-`__index__` arm is
# `'str' object cannot be interpreted as an integer`.
# `W_OSError.descr_set_written` converts through `space.int_w`, which
# names `expected integer, got str object`.
#
# pyre-check: pypy-diverges: pypy3 answers
# `expected integer, got str object`.
e = BlockingIOError()
try:
    e.characters_written = "x"
except TypeError as err:
    assert str(err) == "'str' object cannot be interpreted as an integer", err
else:
    raise AssertionError("expected TypeError")

print("OK")
