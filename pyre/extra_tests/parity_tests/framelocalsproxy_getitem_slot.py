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
cell_and_free_slots()
print("OK")
