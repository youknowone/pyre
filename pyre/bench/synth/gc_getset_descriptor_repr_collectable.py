# pyre-check: no-cpython
import gc


descriptor = type.__dict__["__name__"]
ordinary = repr(descriptor)
direct = type(descriptor).__repr__(descriptor)

assert ordinary == direct
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("getset descriptor repr results are collectable")
