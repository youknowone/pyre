# pyre-check: max-pypy-ratio=7.2
# Run 37874517385, startup-subtracted vs pypy: macos dynasm 1.9x
# (2.1x on its rerun), windows dynasm 2.5x, macos cranelift 2.6x,
# ubuntu dynasm 3.1x, ubuntu cranelift 4.1x. The readings now span 2.16x,
# more than the 2x the old "twice the slowest" rule can hold under a
# floor divisor of 4, so 7.2 keeps the floor 1.8x under the fastest 1.9x
# and the ceiling 1.76x over the slowest 4.1x.
# What this fixture gates is the differential OUTPUT — pyre must agree with
# cpython and pypy that the rebound method wins after the loop compiled.
# `typeobject.py:177 _immutable_fields_ = ['_version_tag?']` — the LOAD_METHOD
# fold bakes the resolved method into the compiled loop under a
# QUASIIMMUT_FIELD on the receiver type's `_version_tag`, with no per-iteration
# read of the tag to re-prove it. Rebinding the method afterwards bumps the tag
# (`mutated()`, typeobject.py:285-286) and must revoke every loop that baked
# the old identity, or the compiled code keeps calling the replaced method.
#
# Second half covers the subclass walk: `mutated()` recurses through
# `weak_subclasses` (typeobject.py:288-291), so rebinding on a BASE has to
# revoke a loop warmed on a subclass instance.
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 3557424


class P:
    def get(self):
        return 1


class Base:
    def val(self):
        return 10


class Sub(Base):
    pass


def run_get(p, n):
    total = 0
    i = 0
    while i < n:
        total = total + p.get()
        i = i + 1
    return total


def run_val(s, n):
    total = 0
    i = 0
    while i < n:
        total = total + s.val()
        i = i + 1
    return total


def run_mutating(q, n):
    """Rebind the method from INSIDE the loop being traced and compiled.

    This is the sharp case: a live read + `guard_value` of the tag would catch a
    mid-trace rebinding on the spot, so under `_version_tag?` the per-iteration
    `GUARD_NOT_INVALIDATED` has to catch it instead. Returns the distinct values
    observed, in order, so a stale trace shows up as a missing transition rather
    than only as a wrong total.
    """
    seen = []
    i = 0
    while i < n:
        if i == n // 8:
            Q.run = lambda self: 3
        if i == n // 2:
            Q.run = lambda self: 9
        v = q.run()
        if not seen or seen[-1] != v:
            seen.append(v)
        i = i + 1
    return seen


class Q:
    def run(self):
        return 1


def main():
    p = P()
    # Phase 1: warm and compile the call loop against the original method.
    a1 = run_get(p, N)
    # Phase 2: rebind on the class itself.
    P.get = lambda self: 2
    a2 = run_get(p, N)
    # Phase 3: rebind again, so a stale second-generation trace is caught too.
    P.get = lambda self: 5
    a3 = run_get(p, N)

    s = Sub()
    b1 = run_val(s, N)
    # Phase 4: rebind on the BASE of the warmed receiver's class.
    Base.val = lambda self: 20
    b2 = run_val(s, N)

    # Phase 5: rebind twice from inside the loop itself.
    c = run_mutating(Q(), N)

    print(a1 // N, a2 // N, a3 // N, b1 // N, b2 // N, c)


main()
