# pyre-check: no-cpython
import gc


descriptor = list.__dict__["append"]
ordinary = repr(descriptor)
direct = type(descriptor).__repr__(descriptor)

assert ordinary == direct
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("method descriptor repr results are collectable")
