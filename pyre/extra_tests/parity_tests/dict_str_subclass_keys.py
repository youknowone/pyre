# CPython-suite gap: dict tests do not cover PyPy MapDict's non-exact-key transition.
# parity-tests reason: the remaining cases target PyPy/pyre MapDict storage only.

"""MapDict transitions must preserve non-exact-str key behavior."""

import sys


class MyStr(str):
    pass


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
if sys.implementation.name == "cpython":
    # CPython 3.14.2's inline-values instance dictionary canonicalizes this
    # equal key into the shared exact-str attribute name and does not expose it
    # through the materialized dict view. PyPy's MapDictStrategy deliberately
    # takes the object-strategy path and preserves the subclass identity.
    assert holder.attr1 == 2
else:
    stored = next(iter(holder.__dict__))
    assert stored is instance_key
    assert type(stored) is MyStr

# The non-str MapDictStrategy leg devolves to ObjectDictStrategy and must keep
# the checked insertion contract across that transition.  Protocol-0 pickle's
# BUILD opcode exposes this by hashing a state key once in the state dict and
# again while assigning it into the new instance's __dict__.
class HashError(Exception):
    pass


class RaisingKey:
    remaining = 1

    def __hash__(self):
        if not self.remaining:
            raise HashError
        self.remaining -= 1
        return 42


raising_key = RaisingKey()
state = {raising_key: None}
try:
    Holder().__dict__[raising_key] = state[raising_key]
except HashError:
    pass
else:
    raise AssertionError("MapDictStrategy swallowed the key's __hash__ error")

print("OK")
