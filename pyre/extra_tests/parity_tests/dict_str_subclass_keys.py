"""Dict strategies must preserve str-subclass key identity.

PyPy's EmptyDictStrategy and UnicodeDictStrategy select their specialized
storage only when ``type(key) is str``.  A str subclass therefore uses the
object strategy so iteration returns the original key object and overridden
hash/equality methods remain observable.
"""


class MyStr(str):
    pass


key = MyStr("attr1")
d = {key: 1}
stored = next(iter(d))
assert stored is key
assert type(stored) is MyStr

# An existing exact-str strategy must de-specialize before accepting a
# subclass key, while preserving insertion order and the subclass object.
d = {"plain": 1}
other = MyStr("attr2")
d[other] = 2
keys = list(d)
assert keys[0] == "plain"
assert keys[1] is other

# Lookup and deletion with a subclass also take the generic comparison path.
probe = MyStr("plain")
assert d[probe] == 1
del d[probe]
assert list(d) == [other]

# MapDictStrategy (the live view behind an instance __dict__) has the same
# exact-str gate.  A shared attribute name from another instance must not
# cause a subclass key to collapse into that plain string.
class Holder:
    pass


seed = Holder()
seed.attr1 = 1
holder = Holder()
instance_key = MyStr("attr1")
holder.__dict__[instance_key] = 2
stored = next(iter(holder.__dict__))
assert stored is instance_key
assert type(stored) is MyStr

print("OK")
