# pyre-check: no-cpython
import gc


union = int | str
ordinary = repr(union)
direct = type(union).__repr__(union)

assert ordinary == direct == "int | str"
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("union repr results are collectable")
