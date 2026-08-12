# CPython-suite gap: test_mmap's exhaustive find/rfind sweep (test_find_end,
# test_rfind) never uses an empty pattern — its list is
# [b"o", b"on", b"two", b"ones", b"s"] — so the near-end answer an empty needle
# owes is untested there. The oversized-needle cases below are covered by that
# sweep, and are kept only to hold the two halves of one bound together.
# parity-tests reason: test_mmap is not in the suite gate (still 3 failures and
# 21 errors), so nothing in the vendored suite protects either half today.

"""`mmap.find`/`rfind` over a span that cannot hold the needle report -1.

The scan's upper bound is `span - len(needle)`. Clamping that subtraction at
zero still leaves one index to try, and reading a needle-sized window there
runs off the end of a shorter span, so the interpreter aborts instead of
answering.

The empty needle is the boundary in the other direction, and it is the half
with no oracle in the vendored suite: it matches at the near end of the span —
`start` for `find`, `end` for `rfind` — where a shared "empty or inverted"
guard would fold it into -1 along with the spans that really have no room.

Every case pins the value rather than the absence of a panic, so the fixture
keeps its meaning once the abort is gone.
"""

import mmap

m = mmap.mmap(-1, 4)
m[:] = b"abca"

# The empty needle matches at the near end of the span.
assert m.find(b"") == 0, m.find(b"")
assert m.rfind(b"") == 4, m.rfind(b"")
assert m.find(b"", 2) == 2, m.find(b"", 2)
assert m.rfind(b"", 0, 2) == 2, m.rfind(b"", 0, 2)
assert m.find(b"", 4) == 4, m.find(b"", 4)
assert m.rfind(b"", 4) == 4, m.rfind(b"", 4)
assert m.find(b"", -2) == 2, m.find(b"", -2)
assert m.rfind(b"", -2) == 4, m.rfind(b"", -2)

# An inverted span holds nothing at all, not even the empty needle.
assert m.find(b"", 3, 1) == -1, m.find(b"", 3, 1)
assert m.rfind(b"", 3, 1) == -1, m.rfind(b"", 3, 1)

# The needle is longer than the whole map.
assert m.find(b"abcab") == -1, m.find(b"abcab")
assert m.rfind(b"abcab") == -1, m.rfind(b"abcab")

# The needle fits the map but not the requested span.
assert m.find(b"abc", 2) == -1, m.find(b"abc", 2)
assert m.rfind(b"abc", 2) == -1, m.rfind(b"abc", 2)
assert m.find(b"abc", 0, 2) == -1, m.find(b"abc", 0, 2)
assert m.rfind(b"abc", 0, 2) == -1, m.rfind(b"abc", 0, 2)

# A span exactly the needle's length still has one candidate.
assert m.find(b"bc", 1, 3) == 1, m.find(b"bc", 1, 3)
assert m.rfind(b"bc", 1, 3) == 1, m.rfind(b"bc", 1, 3)

# The ordinary answers, so a bound that returns -1 too eagerly is caught too.
assert m.find(b"a") == 0, m.find(b"a")
assert m.rfind(b"a") == 3, m.rfind(b"a")
assert m.find(b"ca") == 2, m.find(b"ca")
assert m.find(b"a", -1) == 3, m.find(b"a", -1)
assert m.find(b"abca", -10) == 0, m.find(b"abca", -10)

m.close()

# A one-byte map is the smallest span an oversized needle can overrun.
one = mmap.mmap(-1, 1)
one[:] = b"a"
assert one.find(b"ab") == -1, one.find(b"ab")
assert one.rfind(b"ab") == -1, one.rfind(b"ab")
assert one.find(b"") == 0, one.find(b"")
assert one.rfind(b"") == 1, one.rfind(b"")
assert one.find(b"a") == 0, one.find(b"a")
one.close()

print("OK")
