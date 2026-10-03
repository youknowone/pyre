# CPython-suite gap: `test_exceptions` reads `args` and calls `with_traceback`
# on an exception it just built. It does not reduce that exception afterwards,
# so a lookup that installs an empty instance dict is invisible there.
#
# parity-tests reason: `__reduce__` packs a set instance dict, empty included.
# A method lookup or a missing-name probe that creates that dict changes the
# pickle from a 2-tuple into a 3-tuple whose state is `{}`.
def reduced_len(exc):
    return len(exc.__reduce__())


exc = ValueError(1)
assert getattr(exc, "nope", None) is None
assert reduced_len(exc) == 2, exc.__reduce__()

exc = ValueError(1)
assert hasattr(exc, "nope") is False
assert reduced_len(exc) == 2, exc.__reduce__()

exc = ValueError(1)
assert getattr(exc, "\udc80", None) is None
assert reduced_len(exc) == 2, exc.__reduce__()

exc = ValueError(1)
assert hasattr(exc, "\udc80") is False
assert reduced_len(exc) == 2, exc.__reduce__()

exc = OSError(2, "m", "a", 5, "b")
assert getattr(exc, "winerror", "NOWIN") == "NOWIN"
assert reduced_len(exc) == 2, exc.__reduce__()

exc = ValueError(1)
assert getattr(exc, "__notes__", None) is None
assert reduced_len(exc) == 2, exc.__reduce__()

exc = ValueError(1)
bound = exc.with_traceback
assert reduced_len(exc) == 2, exc.__reduce__()
assert bound(None) is exc
assert exc.__traceback__ is None
assert reduced_len(exc) == 2, exc.__reduce__()

exc = ValueError(1)
assert hasattr(exc, "with_traceback") is True
assert reduced_len(exc) == 2, exc.__reduce__()

exc = ValueError(1)
assert exc.args == (1,)
assert reduced_len(exc) == 2, exc.__reduce__()


class E(ValueError):
    pass


exc = E(1)
assert exc.with_traceback(None) is exc
assert reduced_len(exc) == 2, exc.__reduce__()

exc = ValueError(1)
exc.k = 1
assert getattr(exc, "nope", None) is None
assert exc.k == 1
assert exc.__dict__["k"] == 1

exc = ValueError(1)
setattr(exc, "\udc80", 7)
assert getattr(exc, "\udc80") == 7
assert reduced_len(exc) == 3, exc.__reduce__()

exc = ValueError(1)
exc.with_traceback = "shadow"
assert exc.with_traceback == "shadow"
assert exc.__dict__["with_traceback"] == "shadow"


class D(dict):
    pass


exc = ValueError(1)
holder = D(k=3)
exc.__dict__ = holder
assert exc.k == 3
assert exc.__dict__ is holder

print("OK")
