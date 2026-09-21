# pyre-check: gate=1
"""The class `gc.get_stats()` returns keeps working across a collection.

`app_referents.GcStats` is built on the first call and named by no module
dict, so between one returned instance and the next its only owner is the
collector's own root walk.  Without that root a major collection sweeps the
class, and the next call mints an instance over freed memory: the first
attribute store walks the instance's MRO and reads a word that is no longer
a type.

Live-then-dead payload is what carries a major collection through its sweep,
so the churn below is the whole of the setup.  The reference build's
`get_stats()` is a list and has no such class to lose; there the arm that
reads a field does not apply.
"""
import gc

MB = 1024 * 1024
COUNT = 64


def the_stats_class_outlives_a_major_collection():
    gc.get_stats()
    junk = [bytes(MB) for _ in range(COUNT)]
    del junk
    gc.collect()
    gc.collect()
    total = getattr(gc.get_stats(), 'total_gc_memory', None)
    if total is None:  # the reference build's `get_stats()` is a list
        return
    assert total, total


the_stats_class_outlives_a_major_collection()
print('OK')
