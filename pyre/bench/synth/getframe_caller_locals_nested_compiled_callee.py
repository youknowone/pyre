# Regression guard: a callee that reads its CALLER's frame via sys._getframe(1)
# must see the caller's live locals, even though the callee has a hot loop of its
# own and so takes its own compiled-loop entry from inside the caller's running
# compiled loop.
#
# Two defects met here and produced a JIT-only wrong answer (acc frozen at the
# value it held when the caller's loop was compiled):
#   1. the force was skipped, because it required the frame to equal
#      `MetaInterp::vable_ptr` and the callee's compiled entry had re-pointed
#      that cell at the callee frame with no restore;
#   2. once the force ran it wrote a null over `acc`, because the resume decode
#      went through an allocator that could not materialize the virtual the
#      slot named -- and a null slot reads back as an ABSENT name, so
#      `f_locals['acc']` raised KeyError instead of returning a value.
import sys


def inner(k):
    s = 0
    for j in range(3):        # callee's own hot loop -> its own compiled entry
        s += j * k
    if k > 29990:
        g = sys._getframe(1)  # the CALLER's frame, mid-activation
        return s + g.f_locals['acc']
    return s


def outer(n):
    acc = 0                   # loop-carried, virtual inside the compiled loop
    for i in range(n):
        acc += inner(i) & 7
    return acc


print(outer(30000))
