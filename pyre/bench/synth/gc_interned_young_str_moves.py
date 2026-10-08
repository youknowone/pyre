# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# pyre-check: requires-modules=gc
import gc
import sys

# `ObjSpace.interned_strings` is a weak-value dictionary
# (`make_weak_value_dictionary` / `RWeakValueDictionary`,
# `test_interned_strings_are_weak`). `sys.intern` of a string built at run time
# stores a young WEAKREF into the table: the table write barriers the table
# object (`ll_set_nonnull`), a minor collection copies the WEAKREF and
# `invalidate_young_weakrefs` rewrites `weakptr` to the copied string. The
# interned strings are not extra roots.
#
# Each round interns fresh strings, churns the nursery so at least one minor
# collection runs, and re-interns equal values built from new characters:
# while a reference is held, the table must return the held object.
#
# The held strings are dict values under int keys. A list of exact ascii `str`
# takes `AsciiListStrategy`, which stores only the utf8 payload and wraps a new
# object on read (`AsciiListStrategy.unwrap`/`wrap`), so the interned object
# itself would not be the one held.
#
# Self-checking.

kept = []
for r in range(30):
    fresh = {i: sys.intern("interned-%d-%d" % (r, i)) for i in range(50)}
    kept.append(fresh)
    for _ in range(4):
        junk = [[j] for j in range(3000)]
        del junk
    for i, s in fresh.items():
        again = sys.intern("".join(["interned-", str(r), "-", str(i)]))
        assert again is s, "round %d item %d lost its identity" % (r, i)

gc.collect()
for r, fresh in enumerate(kept):
    for i, s in fresh.items():
        expected = "interned-%d-%d" % (r, i)
        assert s == expected, (s, expected)
        assert sys.intern(expected) is s

# Dropped entries may be collected; re-interning still returns an equal string.
chars = "".join(["interned-weak-", str(id(object()))])
held = sys.intern("".join(list(chars)))
del held, kept, fresh
for _ in range(10):
    gc.collect()
assert sys.intern("".join(list(chars))) == chars

print("PASS")
