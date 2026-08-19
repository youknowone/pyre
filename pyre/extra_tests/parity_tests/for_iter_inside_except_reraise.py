# CPython-suite gap: the suite does not run a hot user-defined iterator to
# exhaustion inside an `except` body and then re-raise the handled exception
# with a bare `raise`.
# parity-tests reason: FOR_ITER consumes a StopIteration, and the frame state
# on the consuming edge must keep naming the exception the enclosing handler
# caught -- otherwise the bare `raise` re-raises the loop's own exhaustion
# signal instead of the handled exception.

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


class It:
    """A user-defined iterator, so FOR_ITER takes the instance `__next__` route."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n


def bare_reraise(n):
    """The bare `raise` sits in the handler body, uncovered by any other try."""
    try:
        raise ValueError("outer")
    except ValueError:
        for _ in It(n):
            pass
        raise


def bare_reraise_nested(n):
    """Two handlers deep: the innermost handled exception wins."""
    try:
        raise KeyError("outer")
    except KeyError:
        try:
            raise IndexError("inner")
        except IndexError:
            for _ in It(n):
                pass
            raise


def reraise_by_name(n):
    """The named-exception control: `raise e` never reads the frame's pair."""
    try:
        raise ValueError("outer")
    except ValueError as e:
        for _ in It(n):
            pass
        raise e


def propagate_from_handler(n):
    """A fresh raise out of the handler chains the handled exception as context."""
    try:
        raise ValueError("outer")
    except ValueError:
        for _ in It(n):
            pass
        raise TypeError("fresh")


def caught(fn, n):
    try:
        fn(n)
    except BaseException as e:  # noqa: BLE001 - the defect swaps the exception type
        context = type(e.__context__).__name__ if e.__context__ is not None else None
        return (type(e).__name__, str(e), context)
    return ("no exception", "", None)


for _ in range(3000):
    # An empty loop body and a non-empty one reach the exhaustion edge with
    # different stack depths, so exercise both.
    for count in (0, 3):
        assert caught(bare_reraise, count) == ("ValueError", "outer", None)
        assert caught(bare_reraise_nested, count) == ("IndexError", "inner", "KeyError")
        assert caught(reraise_by_name, count) == ("ValueError", "outer", None)
        assert caught(propagate_from_handler, count) == (
            "TypeError",
            "fresh",
            "ValueError",
        )

print("OK")
