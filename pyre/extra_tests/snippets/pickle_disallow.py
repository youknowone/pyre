# `Py_TPFLAGS_DISALLOW_INSTANTIATION` (1 << 7): a generator's type has a
# NULL `tp_new`, so `generator()` raises `cannot create 'generator'
# instances` and pickling raises `cannot pickle 'generator' object` at
# every protocol.  A generator is only ever produced by calling a
# generator function.
import io
import pickle
import _pickle

DISALLOW = 1 << 7


def gen():
    yield 1


g = gen()
G = type(g)

# The flag is exposed through `__flags__`.
assert G.__flags__ & DISALLOW, hex(G.__flags__)

# Direct instantiation is refused.
try:
    G()
except TypeError as e:
    assert str(e) == "cannot create 'generator' instances", str(e)
else:
    raise AssertionError("generator() should raise")

# Pickling is refused at every protocol, through both the bound
# `__reduce_ex__` and the full pickler paths.
for proto in range(0, pickle.HIGHEST_PROTOCOL + 1):
    try:
        gen().__reduce_ex__(proto)
    except TypeError as e:
        assert str(e) == "cannot pickle 'generator' object", str(e)
    else:
        raise AssertionError(("__reduce_ex__ should raise", proto))

try:
    pickle.dumps(gen())
except TypeError as e:
    assert str(e) == "cannot pickle 'generator' object", str(e)
else:
    raise AssertionError("pickle.dumps(gen) should raise")

try:
    buf = io.BytesIO()
    _pickle.Pickler(buf, 2).dump(gen())
except TypeError as e:
    assert str(e) == "cannot pickle 'generator' object", str(e)
else:
    raise AssertionError("_pickle.Pickler.dump(gen) should raise")

# The flag does not leak: a generator still iterates, and ordinary
# classes remain instantiable and picklable.
assert list(gen()) == [1]


class C:
    def __init__(self):
        self.x = 1


assert not (C.__flags__ & DISALLOW)
assert C().x == 1
assert pickle.loads(pickle.dumps(C())).x == 1

print("pickle_disallow OK")
