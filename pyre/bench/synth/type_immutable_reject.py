# pyre-check: max-pypy-ratio=40
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio observed
# across CI hosts (18.2x), rounded up, and no floor gate is recorded.
N = 200000


# type.__setattr__ / type.__delattr__ reject mutation of a non-heap
# (immutable) builtin type with TypeError before touching the type dict
# (typeobject.py setdictvalue/deldictvalue heaptype guard).  The raising
# STORE_ATTR / DELETE_ATTR runs every iteration, so the JIT records a
# GuardNoException after the residual store and deopts into the blackhole,
# which must resume at the loop's handler.
def main():
    acc = 0
    i = 0
    while i < N:
        try:
            int.injected = i
        except TypeError:
            acc = acc + 1
        try:
            del str.injected
        except TypeError:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
