# pyre-check: max-pypy-ratio=41
# PEP 479: a StopIteration that escapes a generator body is replaced by
# RuntimeError("generator raised StopIteration") chained from it, across every
# drive path (list / for / tuple / next / send) and whether the StopIteration
# was raised explicitly or leaked from a `next()` in the body.  A normal
# generator return is unaffected.  3.14 and PyPy agree, so check.py (PyPy
# oracle) can assert this.


def g_raise():
    yield 1
    raise StopIteration


def g_nextleak():
    yield 1
    next(iter([]))


def g_return():
    yield 1
    return 7


def drive():
    out = []

    def check(label, fn) -> None:
        try:
            out.append((label, "ok", fn()))
        except RuntimeError as e:
            cause = type(e.__cause__).__name__ if e.__cause__ is not None else None
            out.append((label, "runtimeerror", str(e), cause))
        except Exception as e:  # noqa: BLE001
            out.append((label, "other", type(e).__name__))

    check("list", lambda: list(g_raise()))
    check("for", lambda: [x for x in g_raise()])  # noqa: C416 - exercises the for-loop iteration path distinctly from the list() case
    check("tuple", lambda: tuple(g_raise()))
    check("next", lambda: (lambda it: [next(it), next(it)])(g_raise()))
    check("send", lambda: (lambda it: (next(it), it.send(None)))(g_raise()))
    check("nextleak", lambda: list(g_nextleak()))
    check("normal_return", lambda: list(g_return()))

    # Drive the conversion hot so a compiled trace exercises the leak path.
    hot = 0
    n = 0
    while n < 20000:
        try:
            list(g_raise())
        except RuntimeError:
            hot += 1
        n += 1
    out.append(("hot", hot))

    return out


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


for row in drive():
    print(row)
subclass_paths()
