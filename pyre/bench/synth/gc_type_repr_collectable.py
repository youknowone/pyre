# pyre-check: no-cpython
import gc


class Sample:
    pass


ordinary = repr(Sample)
direct = type.__repr__(Sample)

assert ordinary == direct
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("type repr results are collectable")
