# pyre-check: no-cpython
import gc


functions = (len, [].append)
for function in functions:
    ordinary = repr(function)
    direct = type(function).__repr__(function)
    assert ordinary == direct
    assert any(obj is ordinary for obj in gc.get_objects())
    assert any(obj is direct for obj in gc.get_objects())

print("builtin function repr results are collectable")
