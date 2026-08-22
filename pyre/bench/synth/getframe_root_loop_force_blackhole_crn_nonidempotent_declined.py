# Historical `_declined` companion to the non-idempotent blackhole CRN
# regression.  Exact portal `_getframe(0).f_locals` is now force-free, so this
# upstream shape stays in the compiled loop and never enters that handoff.
#
# PyPy reports one loop, no bridge, no forcings, no virtualizable forcings, and
# no aborts.  The list append remains a strong observable guard: eliminating
# the unnecessary force must still append exactly once per Python iteration.
import sys


def main():
    total = 0
    seen = []
    for i in range(20000):
        fr = sys._getframe(0)
        seen.append(len(fr.f_locals))
        total += i
    print(total, len(seen))


main()
