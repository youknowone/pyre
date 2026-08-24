# pyre-check: gate=1
"""A live `bytes` payload is memory the collector knows it is holding.

`bytesobject.py W_BytesObject._value` is an RPython string -- a varsize GC
object whose characters live inside it -- so the characters are part of the
heap `get_total_memory_used()` measures, and they move the major-collection
threshold like anything else.  Holding the payload in a malloc'd buffer behind
a pointer instead makes the collector's accounting say a few dozen bytes per
object, and a program whose live set is mostly `bytes` then never reaches a
threshold that would collect it.

`gc.get_stats().total_gc_memory` is that accounting, so the difference across a
block of live `bytes` is the whole assertion.  It reads as a formatted size
rather than a count of bytes, which is what PyPy spells too, so the arm parses
it.  The reference build reports its GC statistics as a list and has no such
field; there the arm does not apply.
"""
import gc

MB = 1024 * 1024
COUNT = 64

UNITS = {'kB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3}


def total_gc_memory():
    """`total_gc_memory` in bytes, or None where the field does not exist."""
    text = getattr(gc.get_stats(), 'total_gc_memory', None)
    if text is None:
        return None
    for suffix, scale in UNITS.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * scale)
    return int(float(text))


def a_live_bytes_payload_is_counted():
    before = total_gc_memory()
    if before is None:  # the reference build's `get_stats()` is a list
        return
    hold = [bytes(MB) for _ in range(COUNT)]
    after = total_gc_memory()
    # Keep the payload live across the read, and touch it so nothing can argue
    # the allocation was elided.
    assert len(hold) == COUNT and hold[-1][MB - 1] == 0
    grew = after - before
    # Allow for what the run already held and for allocator rounding; the
    # question is whether the payload is counted at all, and the failing shape
    # answers with a few dozen bytes per object rather than a megabyte.
    assert grew >= COUNT * MB * 3 // 4, (grew, COUNT * MB)
    del hold
    gc.collect()


a_live_bytes_payload_is_counted()
print('OK')
