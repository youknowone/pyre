# CPython-suite gap: `test_exceptions` pickles an `OSError` built by the
# constructor. It does not replace `args` afterwards, and it does not delete
# `filename`.
#
# parity-tests reason: `__reduce__` is the constructor call pickle will replay.
# Reattaching a filename after `args` has stopped being the trimmed pair stores
# different arguments. Deleting the member clears the slot, so the message and
# the rebuilt arguments both drop it.
#
# pyre-check: pypy-diverges: `W_OSError.descr_reduce` appends `w_filename`
# whenever the slot is set, so a replaced `args` tuple grows.
# `readwrite_attrproperty_w` installs no `fdel`, so `del exc.filename` raises
# AttributeError there. `OSError_reduce` splices only when `args` still has
# length 2, and `PyMember_SetOne` clears the member.
import sys


def rebuilt(exc):
    return exc.__reduce__()[1]


exc = OSError(2, "m", "a")
exc.args = (9, "z", "nope")
assert rebuilt(exc) == (9, "z", "nope"), rebuilt(exc)
assert str(exc) == "[Errno 2] m: 'a'", str(exc)

exc = FileNotFoundError(2, "m", "a")
exc.args = (1,)
assert exc.__reduce__()[0] is FileNotFoundError
assert rebuilt(exc) == (1,), rebuilt(exc)
assert str(exc) == "[Errno 2] m: 'a'", str(exc)

exc = OSError()
exc.filename = "a"
assert str(exc) == "[Errno None] None: 'a'", str(exc)
assert rebuilt(exc) == (), rebuilt(exc)
exc.filename2 = "b"
assert str(exc) == "[Errno None] None: 'a' -> 'b'", str(exc)
assert rebuilt(exc) == (), rebuilt(exc)

exc = OSError(2, "m", "a")
del exc.filename
assert exc.filename is None
assert str(exc) == "[Errno 2] m", str(exc)
assert rebuilt(exc) == (2, "m"), rebuilt(exc)

exc = OSError(2, "m", "a", None, "b")
del exc.filename2
assert exc.filename == "a"
assert exc.filename2 is None
assert rebuilt(exc) == (2, "m", "a"), rebuilt(exc)
if sys.platform != "win32":
    assert str(exc) == "[Errno 2] m: 'a'", str(exc)

print("OK")
