# CPython-suite gap: the suite leaks StopIteration itself out of a generator
# but never a subclass of it, so a flat exception-tag test passes the suite.
# parity-tests reason: PEP 479 conversion selects on the exception MRO, so a
# StopIteration subclass leaking a generator must become RuntimeError exactly
# as its base does -- including under multiple inheritance, where the tag of
# the first base decides nothing.

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


class Sub(StopIteration):
    pass


class SV(StopIteration, ValueError):
    pass


class VS(ValueError, StopIteration):
    pass


class SubAsync(StopAsyncIteration):
    pass


class VSA(ValueError, StopAsyncIteration):
    pass


class Exhausted:
    """An iterator whose exhaustion signal is a StopIteration subclass."""

    def __init__(self, stop_type):
        self.stop_type = stop_type

    def __iter__(self):
        return self

    def __next__(self):
        raise self.stop_type("inner")


def g_raise(stop_type):
    yield 1
    raise stop_type("boom")


def g_leak(stop_type):
    yield 1
    next(Exhausted(stop_type))


def drive_coroutine(coro):
    """Step a coroutine to completion without an event loop."""
    try:
        while True:
            coro.send(None)
    except StopIteration as e:
        return e.value


def classify(fn):
    """Run `fn`, reporting how the generator's escaping exception surfaced."""
    try:
        return ("ok", fn())
    except RuntimeError as e:
        cause = e.__cause__
        return ("runtimeerror", str(e), type(cause).__name__)
    except BaseException as e:  # noqa: BLE001 - the defect is a leak, so catch it
        return ("leaked", type(e).__name__)


def exercise(stop_type):
    name = stop_type.__name__
    for label, gen in (("raise", g_raise), ("leak", g_leak)):
        for drive in (
            lambda g: list(g),
            lambda g: [x for x in g],  # noqa: C416 - the for loop is its own path
            lambda g: tuple(g),
            lambda g: (next(g), next(g)),
            lambda g: (g.send(None), g.send(None)),
        ):
            got = classify(lambda: drive(gen(stop_type)))
            assert got == (
                "runtimeerror",
                "generator raised StopIteration",
                name,
            ), (label, name, got)


def exercise_async(stop_type, expected_message):
    async def ag():
        yield 1
        raise stop_type("boom")

    it = ag()
    assert drive_coroutine(it.__anext__()) == 1
    got = classify(lambda: drive_coroutine(it.__anext__()))
    assert got == ("runtimeerror", expected_message, stop_type.__name__), (
        stop_type.__name__,
        got,
    )


def exercise_return():
    """A generator that simply returns is untouched by the conversion."""

    def g():
        yield 1
        return 7

    assert list(g()) == [1]


for _ in range(2000):
    exercise(StopIteration)
    exercise(Sub)
    exercise(SV)
    exercise(VS)
    exercise_async(StopAsyncIteration, "async generator raised StopAsyncIteration")
    exercise_async(SubAsync, "async generator raised StopAsyncIteration")
    exercise_async(VSA, "async generator raised StopAsyncIteration")
    exercise_async(Sub, "async generator raised StopIteration")
    exercise_async(VS, "async generator raised StopIteration")
    exercise_return()

print("OK")
