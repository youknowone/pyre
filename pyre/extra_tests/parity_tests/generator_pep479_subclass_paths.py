# CPython-suite gap: generator tests do not hot-loop PEP 479 over mixed-MRO stop subclasses.
# parity-tests reason: pins generator and async-generator subclass translation off the synth gate.

"""PEP 479 converts StopIteration subclasses in every relevant MRO order."""

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
    def __init__(self, stop_type):
        self.stop_type = stop_type

    def __iter__(self):
        return self

    def __next__(self):
        raise self.stop_type


def g_subclass(stop_type, leak):
    yield 1
    if leak:
        next(Exhausted(stop_type))
    raise stop_type


async def ag_subclass(stop_type):
    yield 1
    raise stop_type


def finish(awaitable):
    try:
        while True:
            awaitable.send(None)
    except StopIteration as stop:
        return stop.value


def subclass_paths():
    for _ in range(2000):
        # Consumer spellings are covered above; this loop owns MRO selection.
        for stop_type in (Sub, SV, VS):
            for leak in (False, True):
                try:
                    list(g_subclass(stop_type, leak))
                except RuntimeError as exc:
                    assert type(exc.__cause__) is stop_type
                    assert str(exc) == "generator raised StopIteration"
                else:
                    raise AssertionError("StopIteration subclass escaped")
        for stop_type, message in (
            (StopAsyncIteration, "async generator raised StopAsyncIteration"),
            (SubAsync, "async generator raised StopAsyncIteration"),
            (VSA, "async generator raised StopAsyncIteration"),
            (Sub, "async generator raised StopIteration"),
            (VS, "async generator raised StopIteration"),
        ):
            iterator = ag_subclass(stop_type)
            assert finish(iterator.__anext__()) == 1
            try:
                finish(iterator.__anext__())
            except RuntimeError as exc:
                assert type(exc.__cause__) is stop_type
                assert str(exc) == message
            else:
                raise AssertionError("async generator stop escaped")


subclass_paths()
print("OK")
