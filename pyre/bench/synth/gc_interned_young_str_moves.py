# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# `sys.intern` of a string built at run time stores a nursery object into the
# process-wide intern table. `intern_exact_str` registers a root for a string
# the collector owns, so a minor collection rewrites the table slot when it
# copies the string. A slot left unrooted would keep naming the old nursery
# address after the copy: the next `sys.intern` of an equal value would then
# return a dead object, and `is` against the survivor would fail.
#
# Each round interns fresh strings, churns the nursery so at least one minor
# collection runs, and re-interns equal values built from new characters.
#
# Self-checking because pypy3 (3.11) does not keep this identity:
# `ObjSpace.interned_strings` is a weak-value dictionary there, and the same
# loop loses every entry of some rounds with the JIT on or off.  CPython 3.14
# keeps it.

import gc
import sys


kept = []
for r in range(30):
    fresh = [sys.intern("interned-%d-%d" % (r, i)) for i in range(50)]
    kept.extend(fresh)
    for _ in range(4):
        junk = [[j] for j in range(3000)]
        del junk
    for i, s in enumerate(fresh):
        again = sys.intern("".join(["interned-", str(r), "-", str(i)]))
        assert again is s, "round %d item %d lost its identity" % (r, i)

gc.collect()
for n, s in enumerate(kept):
    r, i = divmod(n, 50)
    expected = "interned-%d-%d" % (r, i)
    assert s == expected, (s, expected)
    assert sys.intern(expected) is s

print("PASS")
