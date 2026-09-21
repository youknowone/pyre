# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=run,entry-bridge:finish_gen,entry-bridge:finish_awaitable
# A generator or async generator that raises something other than
# StopIteration (ValueError, or the RuntimeError PEP 479 substitutes for a
# StopIteration subclass / StopAsyncIteration) must propagate out of
# `finish()`'s `except StopIteration as stop: return stop.value`.
# Compiled `finish()` guards the recorded exception after the `next()` /
# `send()` call, so a different exception fails that guard. The failure path
# must hand the pending exception on: a GUARD_NO_EXCEPTION carrying the
# propagate-exception descr has to re-raise it, and an exception-guard bridge
# has to be traced with it. Dropping it resumed the frame as if the call had
# returned, and `finish()` returned None / NULL instead of raising.
class Sub(StopIteration):
    pass


def g(stop_type):
    yield 1
    raise stop_type


async def ag(stop_type):
    yield 1
    raise stop_type


def finish_gen(it):
    try:
        while True:
            next(it)
    except StopIteration as stop:
        return stop.value


def finish_awaitable(awaitable):
    try:
        while True:
            awaitable.send(None)
    except StopIteration as stop:
        return stop.value


def run():
    bad_gen = 0
    bad_async = 0
    for _ in range(3000):
        for stop_type in (ValueError, Sub):
            try:
                finish_gen(g(stop_type))
            except (RuntimeError, ValueError):
                pass
            else:
                bad_gen += 1
        for stop_type in (StopAsyncIteration, Sub):
            iterator = ag(stop_type)
            first = finish_awaitable(iterator.__anext__())
            if first != 1:
                bad_async += 1
            try:
                finish_awaitable(iterator.__anext__())
            except RuntimeError:
                pass
            else:
                bad_async += 1
    assert bad_gen == 0 and bad_async == 0, (bad_gen, bad_async)
    print("PASS generator raise through StopIteration handler")


run()
