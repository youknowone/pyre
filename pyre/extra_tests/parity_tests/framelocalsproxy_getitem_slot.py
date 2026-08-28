# CPython-suite gap: `test_frame` reads `frame.f_locals[name]` only for names
# that are bound, so nothing in the suite pins what the proxy reports for an
# unbound slot, for a name it does not carry at all, or for a key that cannot
# be hashed.
#
# `framelocalsproxy_getitem` resolves the key to a locals-plus index and reads
# that one slot.  Answering the same lookup by materializing the whole mapping
# and subscripting it gets the value right and everything else wrong: the miss
# arrives as the mapping's own `KeyError(key)`, and an unhashable key arrives
# as the mapping's key-flavoured `TypeError` rather than the one the hash
# raises before any slot is examined.
#
# The same scan answers a write, and it hashes the key before it looks at any
# name: a key that hashes like nothing the frame carries names no slot, however
# it compares.
#
# parity-tests reason: the value a hit returns is identical either way, so a
# snippet that reads a bound local cannot see the difference.  What separates
# the two shapes is exception text, which is only worth anything next to the
# runtime it has to agree with.
import sys


def scalar_slots():
    bound = 1
    if bound == 0:
        unbound = 2  # noqa: F841 - compiled into a slot that is never bound
    proxy = sys._getframe(0).f_locals
    print("bound", proxy["bound"])
    for name in ("unbound", "absent"):
        try:
            proxy[name]
        except KeyError as exc:
            print(name, "KeyError", exc.args)
    try:
        proxy[["unhashable"]]
    except TypeError as exc:
        print("unhashable", "TypeError", exc)
    # A name the frame has no slot for is stored in, and read back from, the
    # frame's separate extras mapping.
    proxy["extra"] = 7
    print("extra", proxy["extra"])


class SameHash:
    """A non-`str` key that both hashes and compares like the name it holds."""

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return other == self.name

    def __repr__(self):
        # The miss below reports the key with `%R`, so the default repr would
        # put this object's address in the message.
        return f"<key {self.name}>"


class OtherHash(SameHash):
    """The same key, hashing like nothing the frame carries."""

    def __hash__(self):
        return hash(self.name) ^ 1


def non_str_keys():
    bound = 1  # noqa: F841 - read through the proxy below
    proxy = sys._getframe(0).f_locals
    print("same-hash", proxy[SameHash("bound")])
    # The scan compares a name only when its hash matches the key's, so a key
    # that claims equality with a name it does not hash like never reaches the
    # comparison and reads as absent.
    key = OtherHash("bound")
    print("hashes differ", hash(key) != hash(key.name))
    try:
        proxy[key]
    except KeyError as exc:
        print("other-hash", "KeyError", exc.args)


def non_str_key_writes():
    bound = 1
    proxy = sys._getframe(0).f_locals
    # The scan hashes the key before looking at any name, so the key never
    # reaches the extras dict to be reported in its terms.
    try:
        proxy[["unhashable"]] = 1
    except TypeError as exc:
        print("write unhashable", "TypeError", exc)
    proxy[SameHash("bound")] = 2
    print("same-hash write", bound, proxy["bound"])
    # This one is filtered out by the hash before the comparison, so it names
    # no slot and is stored in the extras dict under the key object itself.
    proxy[OtherHash("bound")] = 3
    print("other-hash write", bound, proxy["bound"])
    print("extras keys", sorted(repr(key) for key in proxy if not isinstance(key, str)))


def hidden_slot():
    # PEP 709 inlines a comprehension into its enclosing scope, and in a class
    # body the iteration variable becomes a hidden slot.  Hidden is a property
    # of the write direction only: the scan skips such a slot when it is
    # looking for somewhere to store, so the assignment below goes to the
    # extras dict, but the read that follows still reports the live slot.
    def probe():
        proxy = sys._getframe(1).f_locals
        before = proxy["i"]
        proxy["i"] = 99
        return before, proxy["i"]

    class Body:
        seen = [probe() for i in range(2)]

    print("hidden", Body.seen, "leaked" if "i" in Body.__dict__ else "not leaked")


def cell_and_free_slots():
    captured = "cell"

    def inner():
        own = "own"
        proxy = sys._getframe(0).f_locals
        # `captured` has to be named in this body for the compiler to make it a
        # freevar; reading it only through the proxy would leave it out of the
        # locals-plus table entirely.
        print("free", proxy["captured"], captured, "local", proxy["own"])
        return own

    # `captured` is a varname of this frame AND a cellvar, so the slot holds
    # the cell and a reader has to dereference it; in `inner` the same name is
    # a freevar, which lands past every varname in the locals-plus order.
    print("cell", sys._getframe(0).f_locals["captured"])
    inner()


scalar_slots()
non_str_keys()
non_str_key_writes()
hidden_slot()
cell_and_free_slots()
print("OK")
