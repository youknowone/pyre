"""The memory streams take `seek`'s whence as a C int, its position wider."""

import io


def raises(exc_type, operation):
    try:
        result = operation()
    except exc_type:
        return
    except BaseException as exc:
        raise AssertionError(f"expected {exc_type.__name__}, got {exc!r}") from exc
    raise AssertionError(f"expected {exc_type.__name__}, got {result!r}")


class Index:
    """Only `__index__` — what the whence is read through."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class Both:
    """Both dunders, disagreeing, so which one is consulted is observable."""

    def __index__(self):
        return 0  # SEEK_SET

    def __int__(self):
        return 2  # SEEK_END


class IntOnly:
    def __int__(self):
        return 0


for factory, data in ((io.BytesIO, b"abcdefgh"), (io.StringIO, "abcdefgh")):
    size = len(data)
    stream = factory(data)

    # In range: rejected for what it is, not for how wide it is. The message is
    # the converter's on one side and the range check's on the other, so only
    # the type is compared.
    raises(ValueError, lambda: stream.seek(0, 3))
    raises(ValueError, lambda: stream.seek(0, -1))

    # Out of a C int's range. A whence of 2**32 narrowed to a C int instead
    # would truncate to 0 and seek to the start; taken as a machine int it
    # reaches the range check and comes back as the wrong exception type.
    for whence in (2**32, -(2**32), 2**63, -(2**63) - 1):
        raises(OverflowError, lambda whence=whence: stream.seek(0, whence))

    # The whence is taken through the index protocol. Asserted by value against
    # a non-empty buffer, so SEEK_SET and SEEK_END are distinguishable: reading
    # `Both` through `__int__` would answer `size` here rather than 0.
    assert stream.seek(0, Index(0)) == 0
    assert stream.seek(0, Index(2)) == size
    assert stream.seek(0, Both()) == 0
    raises(OverflowError, lambda: stream.seek(0, Index(2**32)))
    raises(ValueError, lambda: stream.seek(0, Index(3)))

    # `__int__` alone is not the index protocol, so it is not a whence.
    raises(TypeError, lambda: stream.seek(0, IntOnly()))
    raises(TypeError, lambda: stream.seek(0, "0"))
    raises(TypeError, lambda: stream.seek(0, 0.0))

    # The position is a C ssize_t, so one a C int could not hold is a position.
    assert stream.seek(2**32) == 2**32
    raises(OverflowError, lambda: stream.seek(2**63))

print("OK")
