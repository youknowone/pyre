# CPython-suite gap: test_yield_from exercises close() on a generator that
# yields from its finally, but only with locals that hold no finalizer, so the
# prompt-finalization collection this path can take is never entered there.
# parity-tests reason: pyre runs a whole-heap collection at close()'s
# finalization boundary, and that collection may only run once the propagation
# root is consumed; an arm that leaves an exception in flight must skip it.

"""close() propagates its error across the finalization boundary.

The generator's frame holds an object with `__del__`, so releasing the frame
releases a graph that can run application code.  Two of close()'s outcomes
raise rather than return: the generator that yields again after GeneratorExit
becomes RuntimeError, and one that raises from its finally propagates that
exception.  Both must survive the boundary with their message intact.
"""


class Finalized:
    def __del__(self):
        pass


def ignores_generator_exit():
    keeper = Finalized()

    def inner():
        assert keeper is not None
        try:
            yield 1
        finally:
            yield 2

    def outer():
        return (yield from inner())

    return outer()


def raises_from_finally():
    keeper = Finalized()

    def inner():
        assert keeper is not None
        try:
            yield 1
        finally:
            raise ValueError("boom")

    def outer():
        return (yield from inner())

    return outer()


gen = ignores_generator_exit()
assert next(gen) == 1
try:
    gen.close()
except RuntimeError as exc:
    assert str(exc) == "generator ignored GeneratorExit", str(exc)
else:
    raise AssertionError("close() must raise RuntimeError")

# The generator is exhausted, and reading it again must not revisit the
# released graph.
try:
    next(gen)
except StopIteration as exc:
    assert exc.value is None
else:
    raise AssertionError("next() must raise StopIteration")

gen = raises_from_finally()
assert next(gen) == 1
try:
    gen.close()
except ValueError as exc:
    assert str(exc) == "boom", str(exc)
else:
    raise AssertionError("close() must propagate ValueError")

print("OK")
