# pyre-check: gate=1
"""`Pattern.sub` keeps the subject and its buffer current across the filter.

`subx` walks the subject once and calls the replacement per match, so every
turn runs Python.  The subject, the replacement and the subject's buffer object
are read out of the argument slice once, up front; the match object handed to
each call is then stamped with the subject and the buffer, and a match object
is traced.  A stale word here is therefore one the collector follows on its
next walk, not one that is merely read back wrong -- which is why it surfaces
as a crash inside the collector rather than as a wrong substitution.

`count` is resolved through `__index__`, so a collection can happen before the
walk even starts.
"""

import gc
import re

CHURN = None


def collect():
    global CHURN
    gc.collect()
    CHURN = [tuple(range(4)) for _ in range(400)]


pattern = re.compile("b")
SUBJECT = "a" + ("b" + "z") * 40 + "c"
EXPECTED = "a" + ("X" + "z") * 40 + "c"


class Filter:
    """A bound method, which is itself nursery-allocated, writing to its own
    instance dict from inside the walk."""

    def __init__(self):
        self.n = 0

    def repl(self, match):
        self.n += 1
        collect()
        return "X"


f = Filter()
assert pattern.sub(f.repl, SUBJECT) == EXPECTED
assert f.n == 40, f.n

# A second walk over a fresh subject, after the first left its roots behind.
assert len(pattern.split("a" + ("b" + "z") * 40 + "c")) == 41

g = Filter()
out, n = pattern.subn(g.repl, SUBJECT)
assert (out, n) == (EXPECTED, 40), (out, n)


class Count:
    def __index__(self):
        collect()
        return 3


assert pattern.sub("Y", SUBJECT, Count()) == "a" + ("Y" + "z") * 3 + ("b" + "z") * 37 + "c"

# A template replacement takes the other arm, which parses the template after
# the same `__index__` has run.
assert pattern.sub(r"[\g<0>]", SUBJECT, Count()) == (
    "a" + ("[b]" + "z") * 3 + ("b" + "z") * 37 + "c"
)

# A `memoryview` subject is not its own storage: the view's window is gathered
# into a fresh `bytes` and the walk matches against that object's payload, so
# the gathered object -- not the view, and not the backing -- is what has to
# stay rooted across the `__index__` below.  Nothing else refers to it.
bpattern = re.compile(b"b")
BACKING = bytearray(b"a" + (b"b" + b"z") * 40 + b"c")
BYTES_EXPECTED = b"a" + (b"Y" + b"z") * 3 + (b"b" + b"z") * 37 + b"c"

assert bpattern.sub(b"Y", memoryview(BACKING), Count()) == BYTES_EXPECTED
assert bpattern.subn(b"Y", memoryview(BACKING), Count()) == (BYTES_EXPECTED, 3)
# `pos` runs the same `__index__`, and `findall` / `split` keep the gathered
# buffer for the whole walk without a `count` to hang a root on.
assert len(bpattern.findall(memoryview(BACKING), Count())) == 39
assert len(bpattern.split(memoryview(BACKING))) == 41

print("ok")
