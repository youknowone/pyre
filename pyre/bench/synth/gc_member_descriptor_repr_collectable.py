# pyre-check: no-cpython
import gc


class Container:
    __slots__ = ("value",)


descriptor = Container.__dict__["value"]
ordinary = repr(descriptor)
direct = type(descriptor).__repr__(descriptor)

assert ordinary == direct
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("member descriptor repr results are collectable")
