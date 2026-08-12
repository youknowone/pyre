# pyre-check: no-cpython
import gc


def runtime_function():
    pass


ordinary = repr(runtime_function)
direct = type(runtime_function).__repr__(runtime_function)

assert ordinary == direct
assert ordinary.startswith("<function runtime_function at ")
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("function repr results are collectable")
