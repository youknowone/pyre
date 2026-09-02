# pyre-check: max-pypy-ratio=16
# A CALL_KW directly in the hot loop body (not nested inside an inlined
# callee, unlike call_star_forms_inlined_callee.py).  `g(i, step=2)` lowers to
# CALL_KW whose `null_or_self` receiver slot (arg index 1) is the PY_NULL
# sentinel (GcRef(0)) for a plain no-receiver call.  The residual-executor
# NULL-Ref-arg refusal carries the `is_call_kw` receiver-slot exemption its
# sibling walker_abort_if_mayforce_null_ref_arg has.  Without it the refusal
# declines the recording iteration's call to a symbolic op and drops that
# iteration's effect, leaving the hot-loop sum exactly one term short.
# The keyword overrides a default (`step=1`) and the result depends on both
# the positional and the keyword arg so the call cannot be constant-folded
# away; the exact aggregate makes a single dropped iteration observable.
# Sized for the dropped-iteration check, not for an armed ratio gate.  Arming
# one needs a pypy execution over the floor-gate minimum, which this loop
# reaches only past fifty million iterations, and pyre walks the same
# iterations far more slowly here than it does with the keyword spelled
# positionally -- so a length that arms the gate is also one that exceeds
# the per-fixture timeout.  The keyword-call cost is the thing to fix.
N = 400000


def g(x, step=1):
    return x + step


def main():
    total = 0
    for i in range(N):
        total += g(i, step=2)
    print(total)


main()
