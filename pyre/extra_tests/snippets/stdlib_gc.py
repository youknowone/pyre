import gc

from testutils import assert_raises


assert isinstance(gc.collect(), int)
assert isinstance(gc.collect(0), int)
assert isinstance(gc.collect(1), int)
assert isinstance(gc.collect(2), int)
assert isinstance(gc.collect(generation=2), int)


class Index:
    def __index__(self):
        return 0


assert isinstance(gc.collect(Index()), int)

for generation in (-1, 3):
    with assert_raises(ValueError):
        gc.collect(generation)

with assert_raises(TypeError):
    gc.collect("0")

with assert_raises(TypeError):
    gc.collect(0, 1)

assert isinstance(gc.get_objects(), list)
assert isinstance(gc.get_objects(None), list)
assert isinstance(gc.get_objects(generation=None), list)
assert isinstance(gc.get_objects(-1), list)
assert isinstance(gc.get_objects(0), list)
assert gc.get_objects(1) == []
assert isinstance(gc.get_objects(2), list)
assert isinstance(gc.get_objects(Index()), list)

marker = []
assert any(obj is marker for obj in gc.get_objects())
assert any(
    obj is marker
    for obj in gc.get_objects(0) + gc.get_objects(2)
)

for generation in (-2, 3):
    with assert_raises(ValueError):
        gc.get_objects(generation)

for generation in ("0", 1.25):
    with assert_raises(TypeError):
        gc.get_objects(generation)

with assert_raises(TypeError):
    gc.get_objects(None, None)
