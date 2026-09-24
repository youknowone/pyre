# CPython-suite gap: async-generator tests do not hot-loop a one-shot asend.
# parity-tests reason: a redirected CALL_ASSEMBLER guard must not send twice.

"""One `asend`/`__anext__` awaitable is sent exactly once."""


class Sub(StopIteration):
    pass


async def ag(stop_type):
    yield 1
    raise stop_type


def finish(awaitable):
    try:
        while True:
            awaitable.send(None)
    except StopIteration as stop:
        return stop.value


def main():
    for _ in range(2500):
        it = ag(Sub)
        assert finish(it.__anext__()) == 1
        try:
            finish(it.__anext__())
        except RuntimeError as exc:
            assert type(exc.__cause__) is Sub
            assert str(exc) == "async generator raised StopIteration"
        else:
            raise AssertionError("async generator stop escaped")
    print("OK")


if __name__ == "__main__":
    main()
