# pyre-check: gate=1
import gc

from testutils import assert_raises


class Index:
    def __index__(self) -> int:
        return 0


# The generation argument is bound and integer-unwrapped, and every value is
# accepted -- `interp_gc.py:7-26 collect` ignores it.  Which generation an
# integer selects is an engineering choice and follows upstream, so no value is
# rejected and `bench/synth/gc_pypy_frontend.py` pins that against the pypy
# oracle.  The return value is a spec axis and is an int.
assert isinstance(gc.collect(), int)
assert isinstance(gc.collect(0), int)
assert isinstance(gc.collect(2), int)
assert isinstance(gc.collect(generation=2), int)
assert isinstance(gc.collect(True), int)
assert isinstance(gc.collect(Index()), int)

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
