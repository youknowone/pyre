# pyre-check: pypy-diverges: `W_OSError.descr_set_written` stores `space.int_w`,
# so an `__int__`-only value is accepted and an oversized int raises
# OverflowError. `OSError_written_set` stores `PyNumber_AsSsize_t`, which
# calls `__index__` and raises ValueError when the int does not fit.
#
# CPython-suite gap: `test_exceptions` assigns an int to `characters_written`
# and reads it back. It does not assign an `__index__` object, an `__int__`
# object, or an int that does not fit a machine word.
#
# parity-tests reason: the setter and `BlockingIOError`'s numeric third
# argument share one conversion. A setter that still calls `int_w` accepts a
# value the constructor rejects, and the overflow type differs from the
# constructor's ValueError.

class IndexOnly:
    def __index__(self):
        return 4


class IntOnly:
    def __int__(self):
        return 3


class Both:
    def __int__(self):
        return 3

    def __index__(self):
        return 4


def assign(exc, value):
    exc.characters_written = value


def rejects(exc, value, exc_type, message):
    try:
        assign(exc, value)
    except exc_type as ex:
        assert str(ex) == message, (type(ex), ex)
    else:
        raise AssertionError(f"accepted {value!r}")


exc = BlockingIOError(11, "m", 1)
assign(exc, IndexOnly())
assert exc.characters_written == 4
assign(exc, Both())
assert exc.characters_written == 4
assign(exc, False)
assert exc.characters_written == 0
rejects(exc, IntOnly(), TypeError, "'IntOnly' object cannot be interpreted as an integer")
rejects(exc, "x", TypeError, "'str' object cannot be interpreted as an integer")
rejects(exc, 1.5, TypeError, "'float' object cannot be interpreted as an integer")
rejects(exc, 2**100, ValueError, "cannot fit 'int' into an index-sized integer")

operand = Both()
built = BlockingIOError(11, "m", operand)
assert built.characters_written == 4, built.characters_written
assert built.args == (11, "m", operand), built.args
assert built.filename is None, built.filename

print("OK")
