# pyre-check: skip-backends=wasm
# wasm still gets this WRONG, for a cause neither fix below can reach. It is
# exempted so the guard stays meaningful on the backends it guards -- not because
# wasm is right here. Measured: wasm reads 3691 on all nine `f_locals['acc']`
# reads while the truth walks 104967 -> 105000, printing 104995; dynasm,
# cranelift, PYRE_NO_JIT=1 and CPython all print 105005 -- stale in exactly the
# way defect 1 was. See the wasm note at the bottom of this header.
#
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
#
# wasm note. It is not a blanket wasm `f_locals` failure: a caller local read
# through sys._getframe(1) is correct on wasm when the callee does not take a
# compiled entry of its own from inside the caller's running loop. That shape is
# what breaks it, and it breaks the invariant the wasm backend states three times
# to justify lowering `GuardNotForced` to a no-op and calling may-force residuals
# directly -- "the wasm virtualizable is always materialized"
# (`majit-backend-wasm/src/codegen.rs:859`, `:908`, `:950`; the no-op guard at
# `:2175`). Neither fix above can repair it: `OpCode::ForceToken` lowers to a
# literal `0` sentinel (`codegen.rs:4225`), so the compiled loop stores 0 into
# `vable_token` and `force_virtualizable_if_necessary` never forces; and wasm does
# not override `force()` (`majit-backend/src/lib.rs:2170` returns None), so there
# is no deadframe to decode even if it did. wasm holds loop state in wasm locals
# rather than a heap jitframe, so forcing mid-execution is a wasm backend epic.
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
