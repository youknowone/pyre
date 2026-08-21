# Historical `_declined` companion to `getframe_root_loop_force_blackhole_crn`.
# Exact portal `_getframe(0).f_locals` now creates its write-through proxy in
# the trace, so this shape never enters the blackhole CRN handoff.
#
# PyPy reports one loop, no bridge, no forcings, no virtualizable forcings, and
# no aborts.  The fixture still pins the result and locals view at a loop back
# edge; its old NULL-slot/blackhole history belongs to the sibling that
# deliberately exercises a real force, not to this upstream-force-free path.
import sys


def main():
    total = 0
    names = set()
    for i in range(20000):
        fr = sys._getframe(0)
        names.add(len(fr.f_locals))
        total += i
    print(total, sorted(names))


main()
