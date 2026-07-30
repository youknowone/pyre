# pyre-check: max-pypy-ratio=50
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
