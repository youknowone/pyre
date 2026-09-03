# pyre-check: gate=1
"""`BaseException.add_note` keeps the note across the attribute operations.

The note is appended last of all.  Before that, `add_note` reads `__notes__`
off the instance, allocates the list when the attribute is absent, and stores
it back -- and both the read and the store dispatch through the instance, so
either can run Python.  The note relocates, so the word `add_note` started
with is the pre-move one, and what lands in the list is whatever took the
freed box: before the fix this body reported

    [<QQQQQQQQ... object at 0xb4e82e958>]

for a note of `'1000000000000000005'`, and the next collection died walking
the list it was stored in (`site=minor_varsize_item_target`,
`holder_type_id=Some(9)`, `holder_offset=Some(8)` -- a list's items).

A `str` literal is a prebuilt code constant and does not move, which is why
every note here is minted by `str(int)` and is wide enough not to be a cached
small-integer digit run.
"""

import gc

KEEP = None


def churn():
    """Take the freed boxes, so a sweep that frees is not invisible."""
    global KEEP
    gc.collect()
    KEEP = [[i] * 24 for i in range(200)] + [bytearray(b"Q" * 96) for _ in range(120)]


class ReadsBack(Exception):
    """`__notes__` is absent on the first call, so the lookup `add_note` makes
    reaches this and collects with the note live."""

    def __getattr__(self, name):
        churn()
        raise AttributeError(name)


class StoresBack(Exception):
    """The store side of the same window."""

    def __setattr__(self, name, value):
        churn()
        object.__setattr__(self, name, value)


# The first mismatch landed on the sixth turn, so one exception is not enough
# to see it; the loop is what walks the note into a nursery block the churn
# above then recycles.
for n in range(40):
    note = str(10**18 + n)
    e = ReadsBack("boom")
    e.add_note(note)
    assert e.__notes__ == [note], (n, e.__notes__)

    # The second note takes the other arm: `__notes__` is present now, so the
    # list is the one already stored rather than a freshly allocated one.
    second = str(10**18 + 500 + n)
    e.add_note(second)
    assert e.__notes__ == [note, second], (n, e.__notes__)

    f = StoresBack("boom")
    note = str(10**18 + 1000 + n)
    f.add_note(note)
    assert f.__notes__ == [note], (n, f.__notes__)

# `add_note` still rejects a non-str, and does so before any of the above.
try:
    ReadsBack("x").add_note(1234)
except TypeError as exc:
    assert "must be str" in str(exc), exc
else:
    raise AssertionError("add_note accepted an int")

print("ok")
