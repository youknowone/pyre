# pyre-check: no-cpython
import gc

wrapper = (1).__add__
ordinary = repr(wrapper)
direct = type(wrapper).__repr__(wrapper)

assert ordinary == direct
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("method-wrapper repr results are collectable")
