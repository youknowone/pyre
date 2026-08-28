# CPython-suite gap: test_contextlib_async drives a hot coroutine, but its
# failure reads as "coroutine did not complete" rather than as a JIT defect.
# parity-tests reason: this guards the RETURN/YIELD discriminator that a
# compiled generator or coroutine frame has to publish for itself.

"""A compiled generator or coroutine frame reports a RETURN, not one more yield."""


def counts(n):
    total = 0
    i = 0
    while i < n:
        total += i
        i += 1
    if n < 0:
        # Never taken; it is what makes this a generator.
        yield 1
    return total


async def counts_async(n):
    total = 0
    i = 0
    while i < n:
        total += i
        i += 1
    return total


def drive(resumable):
    """Resume once and require the frame to run to its return."""
    try:
        resumable.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("the frame yielded its return value instead of returning")


EXPECTED = 19_999_900_000

# The loop runs long enough to compile, so the return is reached from compiled
# code rather than from the interpreter.
assert drive(counts(200_000)) == EXPECTED
assert drive(counts_async(200_000)) == EXPECTED

print("OK")
