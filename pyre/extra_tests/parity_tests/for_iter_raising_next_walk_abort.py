# CPython-suite gap: the suite does not repeatedly trace a FOR_ITER whose
# iterator raises a non-StopIteration exception from __next__.
# parity-tests reason: FOR_ITER's internal catch must forward the materialized
# exception through its match split without aborting the full-body walk and
# dropping an outer-loop iteration.

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

advances = []


class Boom:
    def __iter__(self):
        return self

    def __next__(self):
        advances.append(1)
        return self.missing


def show(x):
    return x


rounds_seen = []
for round_no in range(6):
    advances.clear()
    try:
        show([x for x in Boom()])
    except AttributeError:
        pass
    rounds_seen.append((round_no, len(advances)))

assert rounds_seen == [(i, 1) for i in range(6)], rounds_seen
print("OK")
