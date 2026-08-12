# pyre-check: no-cpython
import gc


descriptor = dict.__dict__["fromkeys"]
ordinary = repr(descriptor)
direct = type(descriptor).__repr__(descriptor)

assert ordinary == direct
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("classmethod descriptor repr results are collectable")
