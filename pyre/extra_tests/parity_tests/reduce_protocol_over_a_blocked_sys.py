# CPython-suite gap: the suite pins the `None` sentinel as an import failure
# (`test_importlib.import_.test_api` `test_blocked_fromlist`) and it exercises
# `object.__reduce_ex__(2)` over slots at length (`test_descr.test_reduce`,
# `test_pickle`), but nothing runs the second while the first is in effect --
# the reduce tests all import from an unmodified `sys.modules`.
#
# parity-tests reason: the protocol-2 reduction is written at app level in both
# pypy and pyre (`objspace/std/objectobject.py`'s `applevel`, which pyre bundles
# as `reduce_protocol_app.py`), and that source opens with `import sys` so it
# can read `sys.modules['copyreg']`.  Bundled interpreter sources are not the
# program's downstream code, so the name they need must not be resolvable
# through the mapping the program just rebound: a runtime that binds it with an
# ordinary import answers the program's sentinel from inside its own reduce
# protocol, and `__reduce_ex__` fails for a reason the program never wrote.
import sys


class Slotted:
    __slots__ = ("a", "b")


obj = Slotted()
obj.a = 1

# Blocking the name is the program's to do; everything downstream of this line
# that imports `sys` is meant to fail.
sys.modules["sys"] = None

# `__reduce_ex__` is not downstream of it.  Protocol 2 reduces through
# `copyreg.__newobj__` with the slots split out of the instance dict, and the
# `None` sentinel above does not reach any of it.
reduction = obj.__reduce_ex__(2)
assert reduction[0].__name__ == "__newobj__", reduction[0]
assert reduction[1] == (Slotted,), reduction[1]
assert reduction[2] == (None, {"a": 1}), reduction[2]

print("OK")
