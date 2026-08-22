# pyre-check: gate=1
"""`gc.set_threshold` stores all three thresholds and `get_threshold` reports them.

3.14's collector is incremental and sizes no third generation, but the value it
is handed still round-trips: `gc_set_threshold_impl` stores `threshold2` and
`gc_get_threshold_impl` builds its tuple from all three slots, in both the
default and the `Py_GIL_DISABLED` arm.  So a caller that saves the thresholds
and restores them gets back what it saved.
"""

import gc

saved = gc.get_threshold()
assert len(saved) == 3, saved
# The values a fresh interpreter starts with, third included.
assert saved == (2000, 10, 10), saved

gc.set_threshold(1, 2, 3)
assert gc.get_threshold() == (1, 2, 3), gc.get_threshold()

# An omitted trailing value keeps the threshold already there.
gc.set_threshold(11, 22)
assert gc.get_threshold() == (11, 22, 3), gc.get_threshold()

gc.set_threshold(44)
assert gc.get_threshold() == (44, 22, 3), gc.get_threshold()

gc.set_threshold(*saved)
assert gc.get_threshold() == saved, (gc.get_threshold(), saved)
