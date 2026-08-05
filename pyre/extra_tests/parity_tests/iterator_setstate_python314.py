"""Cursor restore protocol of the builtin iterators, per Python 3.14.

`__setstate__` is the pickle half of `__reduce__`, so the two have to agree on
which cursor values mean "exhausted".  3.14 answers that per producer: the
length-aware iterators clamp an over-long cursor to the sequence's *current*
length, the generic `__getitem__` iterator cannot and stores it verbatim, and
the two producers that carry exhaustion in the cursor rather than in a cleared
sequence reject every later state once it goes negative.
"""

import array


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def state(it):
    """The cursor `__reduce__` pickles, or EXHAUSTED for the empty form."""
    reduced = it.__reduce__()
    return reduced[2] if len(reduced) > 2 else "EXHAUSTED"


def remaining(it):
    """`__length_hint__`, which `array.arrayiterator` alone does not declare."""
    return it.__length_hint__() if hasattr(it, "__length_hint__") else None


class Getitem:
    """A sequence reachable only through the `__getitem__` protocol."""

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index >= 3:
            raise IndexError(index)
        return index


# ── an over-long cursor clamps to the length, except for the generic iterator ──

for label, make in (
    ("tuple", lambda: iter((10, 20, 30))),
    ("str_ascii", lambda: iter("abc")),
    ("str_wide", lambda: iter("a\xe9中")),
    ("bytes", lambda: iter(b"abc")),
    ("bytearray", lambda: iter(bytearray(b"abc"))),
    ("array", lambda: iter(array.array("i", [10, 20, 30]))),
):
    it = make()
    it.__setstate__(99)
    check(state(it) == 3, label + " clamps an over-long cursor to the length")
    check(remaining(it) in (0, None), label + " reports nothing left at the end")
    check(next(it, "STOP") == "STOP", label + " is exhausted at the length")
    it = make()
    it.__setstate__(2)
    check(state(it) == 2, label + " keeps an in-range cursor")

it = iter(Getitem())
it.__setstate__(99)
check(state(it) == 99, "the generic iterator has no length to clamp against")
check(next(it, "STOP") == "STOP", "an out-of-range generic cursor stops")

# The clamp reads the sequence live, so a mutable producer grown after the
# iterator was made clamps to the new length.
data = bytearray(b"abc")
it = iter(data)
data.extend(b"defghij")
it.__setstate__(99)
check(state(it) == 10, "bytearray clamps against the grown length")

values = [10, 20, 30]
it = iter(values)
values.extend([40, 50])
it.__setstate__(99)
check(state(it) == 5, "list clamps against the grown length")

# ── a negative cursor rewinds, except where it is the exhausted sentinel ──

for label, make in (
    ("tuple", lambda: iter((10, 20, 30))),
    ("str", lambda: iter("abc")),
    ("bytes", lambda: iter(b"abc")),
    ("array", lambda: iter(array.array("i", [10, 20, 30]))),
    ("generic", lambda: iter(Getitem())),
):
    it = make()
    it.__setstate__(-5)
    check(state(it) == 0, label + " rewinds a negative cursor to the front")
    check(remaining(it) in (3, None), label + " has the whole sequence left")

# `list_iterator` and `list_reverseiterator` store -1 as the exhausted cursor
# while keeping the list, so a later in-range state revives them.
it = iter([10, 20, 30])
it.__setstate__(-5)
check(state(it) == "EXHAUSTED", "a negative list cursor is exhausted")
it.__setstate__(1)
check(state(it) == 1 and next(it) == 20, "a list iterator revives")

it = reversed([10, 20, 30])
it.__setstate__(-5)
check(state(it) == "EXHAUSTED", "a negative list_reverseiterator is exhausted")
it.__setstate__(1)
check(state(it) == 1 and next(it) == 20, "a list_reverseiterator revives")

# `bytearray_iterator` and `reversed` reject every later state instead.
for label, make, revived in (
    ("bytearray", lambda: iter(bytearray(b"abc")), 1),
    ("reversed", lambda: reversed((10, 20, 30)), 1),
):
    it = make()
    it.__setstate__(-5)
    check(state(it) == "EXHAUSTED", label + " exhausts on a negative cursor")
    check(remaining(it) == 0, label + " reports nothing left")
    it.__setstate__(revived)
    check(state(it) == "EXHAUSTED", label + " does not revive")
    check(next(it, "STOP") == "STOP", label + " stays exhausted")

# An empty sequence starts `reversed` already off the front.
check(state(reversed(())) == "EXHAUSTED", "reversed(()) starts exhausted")
it = reversed(())
it.__setstate__(0)
check(state(it) == "EXHAUSTED", "reversed(()) cannot be positioned")

# `reversed` clamps an over-long cursor to the last index, not the length.
it = reversed((10, 20, 30))
it.__setstate__(99)
check(state(it) == 2 and next(it) == 30, "reversed clamps to the last index")

# ── the state argument is read as a C ssize_t ──

makers = (
    ("tuple", lambda: iter((10, 20))),
    ("list", lambda: iter([10, 20])),
    ("str", lambda: iter("ab")),
    ("bytes", lambda: iter(b"ab")),
    ("bytearray", lambda: iter(bytearray(b"ab"))),
    ("array", lambda: iter(array.array("i", [10, 20]))),
    ("generic", lambda: iter(Getitem())),
    ("reversed", lambda: reversed((10, 20))),
    ("list_reverse", lambda: reversed([10, 20])),
)


class Index:
    def __index__(self):
        return 1


class MyInt(int):
    pass


for label, make in makers:
    for bad in ("1", 1.5, None, Index()):
        try:
            make().__setstate__(bad)
        except TypeError as exc:
            check(str(exc) == "an integer is required",
                  label + " rejects a non-integer state: " + str(exc))
        else:
            raise AssertionError(label + " accepted a non-integer state")
    try:
        make().__setstate__(1 << 100)
    except OverflowError as exc:
        check(str(exc) == "Python int too large to convert to C ssize_t",
              label + " overflow message: " + str(exc))
    else:
        raise AssertionError(label + " accepted an oversized state")
    it = make()
    it.__setstate__(MyInt(1))
    check(state(it) == 1, label + " accepts an int subclass")

print("OK")
