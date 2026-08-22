# pyre-check: gate=1
"""`gc.set_threshold` stores all three thresholds and `get_threshold` reports them.

3.14's collector is incremental and sizes no third generation, but the value it
is handed still round-trips, so a caller that saves the thresholds and restores
them gets back what it saved.
"""

import gc

saved = gc.get_threshold()
print(len(saved))

gc.set_threshold(1, 2, 3)
print(gc.get_threshold())

# An omitted trailing value keeps the threshold already there.
gc.set_threshold(11, 22)
print(gc.get_threshold())

gc.set_threshold(*saved)
print(gc.get_threshold() == saved)
