# pyre-check: gate=1
# CPython-suite gap: test_mmap reaches find/rfind only through `bytes` and
# `bytearray` patterns, always with a well-formed argument count and never with
# an explicit None bound, so the gateway the two methods share is unpinned
# there apart from the one case where an `__index__` closes the mapping.
# parity-tests reason: test_mmap is not in the suite gate, so nothing in the
# vendored suite protects any of this today.

"""`mmap.find`/`rfind` take their pattern through the buffer protocol.

`view: Py_buffer` is the pattern's converter, so the acquisition -- not a
bytes-only type test -- decides both what counts as a pattern and what a
refusal says.  The export it takes outlives the `start` and `end` conversions,
which run arbitrary `__index__`, so an in-place edit one of them makes is what
the search compares against and a resize one attempts is refused.

The gateway around it orders three refusals: the argument count is checked
before the pattern is acquired, and the pattern before the mapping is checked.
And `end` is converted only when `start` is not None, which is what makes
`m.find(b"Z", None, 5)` a search to the end of the mapping rather than a
rejected None.
"""

import array
import mmap
import sys

DATA = b"a" * 25 + b"Z" + b"b" * 6
LEGACY_CPYTHON = sys.implementation.name == "cpython" and sys.version_info < (3, 14)


def mapping():
    m = mmap.mmap(-1, len(DATA))
    m[:] = DATA
    return m


def raises(kind, message, call, *args):
    try:
        call(*args)
    except kind as error:
        assert str(error) == message, (str(error), message)
    else:
        raise AssertionError("%s%r did not raise %s" % (call, args, kind.__name__))


m = mapping()

# Every exporter is a pattern, not only the two bytes-like builtins.
one = mmap.mmap(-1, 1)
one[0:1] = b"Z"
for pattern in (
    b"Z",
    bytearray(b"Z"),
    memoryview(b"Z"),
    memoryview(b"xZy")[1:2],
    array.array("b", [90]),
    one,
):
    assert m.find(pattern) == 25, pattern
    assert m.rfind(pattern) == 25, pattern
one.close()

# The acquisition names the type it turned down.
for bad in (1, "Z", None, object()):
    message = "a bytes-like object is required, not '%s'" % type(bad).__name__
    raises(TypeError, message, m.find, bad)
    raises(TypeError, message, m.rfind, bad)

# A strided view is refused by the acquisition, which reports why rather than
# naming a wrong argument type.
strided = memoryview(bytearray(b"ZxZx"))[::2]
raises(
    BufferError,
    "memoryview: underlying buffer is not C-contiguous",
    m.find,
    strided,
)
raises(
    BufferError,
    "memoryview: underlying buffer is not C-contiguous",
    m.rfind,
    strided,
)

# CPython moved mmap.find/rfind to Argument Clinic in 3.14.  The generic local
# runner may itself be 3.13, so give that oracle its historical spelling while
# requiring pyre (and 3.14+) to expose the pinned gateway's spelling.
for who in ("find", "rfind"):
    method = getattr(m, who)
    if LEGACY_CPYTHON:
        raises(TypeError, "%s() takes at least 1 argument (0 given)" % who, method)
        raises(
            TypeError,
            "%s() takes at most 3 arguments (4 given)" % who,
            method,
            b"Z",
            0,
            32,
            99,
        )
    else:
        raises(TypeError, "%s expected at least 1 argument, got 0" % who, method)
        raises(
            TypeError,
            "%s expected at most 3 arguments, got 4" % who,
            method,
            b"Z",
            0,
            32,
            99,
        )

# In the 3.14 gateway the argument count is checked before the pattern is
# acquired, and the pattern before the mapping: a closed mapping reports all
# three in that order.  The pre-Clinic CPython gateway checked the mapping
# first, so this pinned-version assertion does not apply to the runner's 3.13.
closed = mapping()
closed.close()
if not LEGACY_CPYTHON:
    for who in ("find", "rfind"):
        method = getattr(closed, who)
        raises(TypeError, "%s expected at least 1 argument, got 0" % who, method)
        raises(
            TypeError,
            "%s expected at most 3 arguments, got 4" % who,
            method,
            b"Z",
            0,
            32,
            99,
        )
        raises(TypeError, "a bytes-like object is required, not 'int'", method, 1)
        raises(ValueError, "mmap closed or invalid", method, b"Z")


class Boom:
    def __index__(self):
        raise AssertionError("an end alongside a None start must not be converted")


# `end` is converted only when `start` is not None, so a None start leaves both
# bounds at the mapping's own and the end argument is never looked at.
if not LEGACY_CPYTHON:
    assert m.find(b"Z", None, 5) == 25, m.find(b"Z", None, 5)
    assert m.rfind(b"Z", None, 5) == 25, m.rfind(b"Z", None, 5)
    assert m.find(b"Z", None, Boom()) == 25
    assert m.rfind(b"Z", None, Boom()) == 25
    assert m.find(b"Z", None, "not an index") == 25

# A None end alongside a real start is the mapping's size, as the default is.
if not LEGACY_CPYTHON:
    assert m.find(b"Z", 0, None) == 25
    assert m.rfind(b"Z", 0, None) == 25
    assert m.find(b"Z", 26, None) == -1

needle = bytearray(b"a")


class EditsPattern:
    def __index__(self):
        needle[0:1] = b"Z"
        return 0


class ResizesPattern:
    def __index__(self):
        needle.extend(b"!")
        return 0


class RaisesFromIndex:
    def __index__(self):
        raise RuntimeError("boom")


# The pattern's bytes are read after the conversions, so the edit is visible.
assert m.find(needle, EditsPattern()) == 25, needle
assert needle == bytearray(b"Z")

# The export is held across them, so the resize is refused -- and released
# afterwards, so the same resize succeeds once the search is over.
raises(
    BufferError,
    "Existing exports of data: object cannot be re-sized",
    m.find,
    needle,
    ResizesPattern(),
)
needle.extend(b"!")
assert needle == bytearray(b"Z!")

# A conversion that raises releases the export too.
raises(RuntimeError, "boom", m.find, needle, RaisesFromIndex())
needle.extend(b"?")
assert needle == bytearray(b"Z!?")

# Both optional conversions finish before the mapping is checked again.  A
# start hook which closes it therefore does not suppress the end hook.
events = []
victim = mapping()


class ClosesMapping:
    def __index__(self):
        events.append("start")
        victim.close()
        return 0


class EndStillRuns:
    def __index__(self):
        events.append("end")
        return len(DATA)


raises(
    ValueError,
    "mmap closed or invalid",
    victim.find,
    b"Z",
    ClosesMapping(),
    EndStillRuns(),
)
assert events == ["start", "end"], events

m.close()

print("OK")
