# `app_abc.py:15-44 SimpleWeakSet`.  The registry and both caches are
# instances of this, so `_get_dump` can hand out the `data` sets and a
# collected entry drops itself through the callback the set installs.
#
# Held at app level rather than rebuilt over the raw set primitives because
# the callback closes over a weakref to the set: the discard has to run with
# the set still reachable but no longer keeping itself alive through it.
from _weakref import ref


class SimpleWeakSet:
    def __init__(self, data=None):
        self.data = set()

        def _remove(item, selfref=ref(self)):
            self = selfref()
            if self is not None:
                self.data.discard(item)

        self._remove = _remove

    def __iter__(self):
        # Weakref callback may remove entry from set.
        # So we make a copy first.
        copy = list(self.data)
        for itemref in copy:
            item = itemref()
            if item is not None:
                yield item

    def __contains__(self, item):
        try:
            wr = ref(item)
        except TypeError:
            return False
        return wr in self.data

    def add(self, item):
        self.data.add(ref(item, self._remove))

    def clear(self):
        self.data.clear()
