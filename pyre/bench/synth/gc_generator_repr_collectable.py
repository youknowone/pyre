# pyre-check: no-cpython
import gc


def generate():
    yield 1


generator = generate()
ordinary = repr(generator)
direct = type(generator).__repr__(generator)

assert ordinary == direct
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

generator.close()

print("generator repr results are collectable")
