# No `max-pypy-ratio`: the only loop here is the two-statement `while` driver,
# so a pypy ratio compares two interpreters' startup rather than any generated
# code, and reads whatever the host's process spawn cost happens to be that
# run. The jitstats baselines gate it.
# Caller locals that are live ACROSS the inlined call, exercised at a vable
# escape inside an inline sub-walk.
#
# The escape is `_gf(1)`, the CALLER's frame, and the depth is load-bearing:
# `try_walker_specialize_sys_getframe` answers depth 0 at the top walk level out
# of the portal virtualizable, so a zero-argument call -- which is what this
# fixture used to carry -- forces nothing and the baselines went to
# `fbw_blackhole_adopted_multi_frame=0`.  Depth 1 names a frame the
# specialization refuses, so the call stays residual and forces.
#
# The multi-frame chain gives each level its own frame as the virtualizable, so
# the walked frame's level writes its `setfield_vable` stores to the LIVE frame,
# while the adopt's resume-state write targets the snapshot and the portal
# epilogue then copies the snapshot's whole locals array back over the live
# frame. Any such store the resume-state write does not cover would revert, and
# `acc` and `tag` below are exactly the kind of caller local that would: both are
# carried across every iteration, so a single reverted slot changes the printed
# totals rather than merely perturbing timing.
import sys

_gf = sys._getframe


def leaf(x):
    _gf(1)
    return x + 1


def main():
    total = 0
    acc = 0
    tag = 7
    i = 0
    while i < 30000:
        total = leaf(total)
        acc = acc + total
        tag = tag ^ i
        i = i + 1
    return total, acc, tag


print(main())
