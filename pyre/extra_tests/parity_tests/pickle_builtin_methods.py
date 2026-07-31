"""Builtin methods and descriptors reduce through getattr(owner, name)."""

import pickle


cases = (
    ("abcd".index, ("c",)),
    (str.index, ("abcd", "c")),
    ([1, 2, 3].__len__, ()),
    (list.__len__, ([1, 2, 3],)),
    ({1, 2}.__contains__, (2,)),
    (set.__contains__, ({1, 2}, 2)),
    (dict.fromkeys, (("a", "b"),)),
    (bytearray.maketrans, (b"abc", b"xyz")),
)

for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    for method, args in cases:
        restored = pickle.loads(pickle.dumps(method, protocol))
        assert restored(*args) == method(*args)

print("OK")
