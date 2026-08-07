# pyre-check: max-pypy-ratio=32
# Companion to getframe_residual_callee_own_frame,
# carrying that file's shape at a force the constant-depth
# `sys._getframe` arm DECLINES, so the machinery it documents stays
# covered now that its own call site folds.
#
# `try_walker_specialize_sys_getframe` (jitcode_dispatch/specialize.rs) takes
# only depth 0 at the top walk level, where `getframe`'s answer IS the portal
# virtualizable and no force is needed.
# The `_getframe` call itself folds here; the added `.f_locals` read is the
# forcing residual, and it forces the SAME frame, so the shape below is
# unchanged apart from where the force comes from.
#
# The counters recorded for this file are the ones its original
# carried before the fold; a diff against them is a real change in the escape
# machinery, not in the arm.
#
# Regression guard: a residual (may-force) callee that inspects its OWN frame
# via sys._getframe() must not clear the traced CALLER's virtualizable tracing
# token. Clearing it raised a spurious frame-escape with no committed resume pc,
# replaying the loop body from entry and double-applying the callee's
# non-journaled STORE_ATTR side effect -- a JIT-only wrong answer (c.n > loops).
import sys


class Counter:
    pass


c = Counter()
c.n = 0


def bump(x):
    c.n += 1                  # STORE_ATTR: non-journaled body effect
    frame = sys._getframe(0)
    frame.f_locals  # may-force residual inspecting the callee's own frame
    return x if frame is not None else -1


def main():
    total = 0
    for i in range(20000):
        total += bump(i)
    print(total, c.n)


main()
