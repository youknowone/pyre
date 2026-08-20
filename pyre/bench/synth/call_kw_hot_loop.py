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
N = 200000


def g(x, step=1):
    return x + step


def main():
    total = 0
    for i in range(N):
        total += g(i, step=2)
    print(total)


main()
