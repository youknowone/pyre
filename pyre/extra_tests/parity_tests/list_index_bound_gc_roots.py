# CPython-suite gap: list.index is never called with an __index__ bound that collects.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.

"""``list.index`` must survive an ``__index__`` bound that moves the list."""

import gc


class Bound:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        # Coercing the start/stop bounds is an arbitrary Python callback, and
        # it runs after `index` has already read its receiver and its needle.
        garbage = [[index] for index in range(20000)]
        assert len(garbage) == 20000
        gc.collect()
        return self.value


class Payload:
    pass


for _ in range(20):
    # A `list` needle exercises the search value on the same window as the
    # receiver: both are kinds whose header a minor collection relocates.
    needle = [Payload()]
    values = [[Payload()] for _ in range(8)]
    values.append(needle)

    assert values.index(needle, Bound(0), Bound(64)) == 8
    assert values.index(needle, Bound(-9)) == 8
    assert values.index(needle, Bound(8), Bound(9)) == 8

print("OK")
