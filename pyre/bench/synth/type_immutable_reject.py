# pyre-check: max-pypy-ratio=40
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio the CI
# runners observe (18.2x), rounded up.
N = 200000


# type.__setattr__ / type.__delattr__ reject mutation of a non-heap
# (immutable) builtin type with TypeError before touching the type dict
# (typeobject.py setdictvalue/deldictvalue heaptype guard).  The raising
# STORE_ATTR / DELETE_ATTR runs every iteration, and
# `try_walker_trace_immutable_type_attr_raise` folds it: the receiver is
# pinned with a GuardValue and the TypeError is emitted as an inline
# NewWithVtable + SetfieldGc construction routed through SubRaise, so both
# the raise and its catch are paid inside the compiled loop.  The recorded
# baselines beside this file read loops_compiled=1, guard_failures=1,
# bridges_compiled=0 — one bailout for the whole run, not one per iteration.
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
