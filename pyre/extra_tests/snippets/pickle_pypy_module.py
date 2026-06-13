from __pypy__ import identity_dict
from __pypy__.builders import BytesBuilder
d = identity_dict()
a = [1, 2]; b = [1, 2]          # equal but distinct objects
d[a] = ("idx_a", a)
assert a in d and b not in d     # identity, not equality
assert d.get(a)[0] == "idx_a"
assert len(d) == 1
bb = BytesBuilder(); bb.append(b"ab"); bb.append(b"cd")
assert bb.build() == b"abcd" and len(bb) == 4

import pickle
for proto in range(0, 6):
    for x in [[1, [2, 3]], {"a": 1, "b": [2]}, {1, 2, 3},
              (1, [2], {3: 4}), [[], {}, ()], {"self": None}]:
        got = pickle.loads(pickle.dumps(x, proto))
        assert got == x, f"proto={proto} {x!r} -> {got!r}"
# recursive structure (memo must handle cycles by identity)
lst = [1]; lst.append(lst)
out = pickle.loads(pickle.dumps(lst))
assert out[0] == 1 and out[1] is out
print("pickle_pypy_module OK")
