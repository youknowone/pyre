# pyre-check: gate=1
# CPython-suite gap: the vendored suites call these entry points with the
# arguments they declare, so nothing in them passes a foreign object to a
# parameter the clinic spells `int`, `double` or `bool`.
# parity-tests reason: `#[pyre_function]` unwrapped a scalar parameter by
# reading the argument's payload at the layout offset for that type, so any
# object at all was accepted and its interior words became the value --
# `msvcrt.SetErrorMode("x")` handed a str's interior to the Win32 call and
# answered with it.

"""A scalar parameter owes a TypeError for an argument of the wrong type."""

import os
import sys
import time


class Foreign:
    """Carries no __index__, __int__, __float__ or __bool__ of its own."""

    __slots__ = ()


def rejects(call, *args):
    try:
        call(*args)
    except TypeError:
        return True
    except OverflowError:
        # A value in range for Python but not for the C type is a different
        # refusal; it is still a refusal, and never a silent read.
        return True
    return False


foreign = Foreign()

# `int` parameters, across the module surface every host carries.
assert rejects(time.sleep, foreign)
assert rejects(os.strerror, foreign)
assert rejects(os.fsync, foreign)
assert rejects(os.close, foreign)
assert rejects(os.dup, foreign)
assert rejects(os.get_inheritable, foreign)
assert rejects(os.set_inheritable, foreign, True)
assert rejects(sys.setrecursionlimit, foreign)

# A float is not an index: it is refused where an integer is asked for, the
# same way `range(1.0)` is.
assert rejects(os.close, 1.5)
assert rejects(sys.setrecursionlimit, 1.5)

# A bool is an int subclass, so it is accepted wherever an integer is.
try:
    os.strerror(True)
except OSError:
    pass

if sys.platform == "win32":
    import msvcrt

    # The value the call answers with is the mode it replaced, so a rejected
    # argument must leave that mode untouched.
    before = msvcrt.GetErrorMode()
    assert rejects(msvcrt.SetErrorMode, foreign)
    assert rejects(msvcrt.SetErrorMode, "x")
    assert rejects(msvcrt.SetErrorMode, [1, 2, 3])
    assert rejects(msvcrt.SetErrorMode, 1.5)
    assert rejects(msvcrt.get_osfhandle, foreign)
    assert rejects(msvcrt.setmode, foreign, foreign)
    assert msvcrt.GetErrorMode() == before, msvcrt.GetErrorMode()

print("OK")
