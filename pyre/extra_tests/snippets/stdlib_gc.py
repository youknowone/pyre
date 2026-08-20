# pyre-check: gate=1
import gc

from testutils import assert_raises


class Index:
    def __index__(self) -> int:
        return 0


# The generation argument is bound and integer-unwrapped, and every value is
# accepted -- `interp_gc.py:7-26 collect` ignores it.  What that argument then
# means, and what `collect` answers, is where pyre follows pypy rather than the
# reference; `bench/synth/gc_pypy_frontend.py` pins both against the pypy
# oracle.  This file asserts only what every implementation agrees on.
gc.collect()
gc.collect(0)
gc.collect(2)
gc.collect(generation=2)
gc.collect(True)
gc.collect(Index())

for generation in (None, 1.25, "0"):
    with assert_raises(TypeError):
        gc.collect(generation)

with assert_raises(TypeError):
    gc.collect(0, 1)


assert isinstance(gc.get_objects(), list)
assert isinstance(gc.get_objects(None), list)
assert isinstance(gc.get_objects(generation=None), list)

marker = []
assert any(obj is marker for obj in gc.get_objects())

with assert_raises(TypeError):
    gc.get_objects(None, None)
