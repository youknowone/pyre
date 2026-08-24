# pyre-check: gate=1
"""A rejected `send` does not start a coroutine, so it still reports as unawaited.

`send(non_None)` on a coroutine that has not run raises before any of its body
executes, so the coroutine is exactly as unawaited afterwards as it was before.
Collecting it therefore still warns.

The reason this needs pinning: `generator.py _invoke_execute_frame` hands the
GC `may_ignore_finalizer` for a started coroutine without `CO_YIELD_INSIDE_TRY`,
and it does so *ahead* of this raise — so the object whose `send` was rejected
carries the hint too, and pypy3 emits no warning here.  3.14 does, so the hint
belongs behind the raise.
"""

import gc
import warnings


async def never_awaited():
    pass


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    coro = never_awaited()
    try:
        coro.send(1)
    except TypeError as exc:
        assert "just-started coroutine" in str(exc), exc
    else:
        raise AssertionError("send(1) on a fresh coroutine did not raise")
    del coro
    gc.collect()
    messages = [str(entry.message) for entry in caught]

assert any("was never awaited" in message for message in messages), messages


# The control: a coroutine that really was started reports nothing, because
# starting it is what settles the question the warning asks.
async def started_then_dropped():
    pass


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    coro = started_then_dropped()
    try:
        coro.send(None)
    except StopIteration:
        pass
    del coro
    gc.collect()
    messages = [str(entry.message) for entry in caught]

assert not any("was never awaited" in message for message in messages), messages
