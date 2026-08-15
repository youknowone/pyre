# CPython-suite gap: exact dict and set __sizeof__ values are not asserted.
# parity-tests reason: guard pyre's CPython 3.14 container size surface.

"""Python 3.14 ``__sizeof__`` surface for dict and set-like types."""

import struct
import sysconfig

for typ in (dict, set, frozenset):
    assert "__sizeof__" in typ.__dict__
    assert typ.__sizeof__.__text_signature__ == "($self, /)"

# The numbers below are the layout of a build that has a global interpreter
# lock. Without one the object header carries the thread id, flags, mutex, gc
# bits and the two refcount halves rather than one refcount, which is two words
# wider, and every container answers that much more. `test.support.calcobjsize`
# derives its struct sizes from the same config var, so the oracle and pyre can
# disagree about the build and still agree about the layout.
HEADER = 2 * struct.calcsize("P") if sysconfig.get_config_var("Py_GIL_DISABLED") else 0

assert dict().__sizeof__() == 48 + HEADER
assert {0: None}.__sizeof__() == 208 + HEADER
assert {str(i): None for i in range(6)}.__sizeof__() == 256 + HEADER
assert dict.fromkeys(range(11)).__sizeof__() == 616 + HEADER

for typ in (set, frozenset):
    assert typ().__sizeof__() == 200 + HEADER
    assert typ(range(4)).__sizeof__() == 200 + HEADER
    assert typ(range(5)).__sizeof__() == 712 + HEADER
    assert typ(range(19)).__sizeof__() == 2248 + HEADER

print("OK")
