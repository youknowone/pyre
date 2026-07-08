# A CALL_KW directly in the hot loop body (not nested inside an inlined
# callee, unlike call_kw_star.py).  `g(i, step=2)` lowers to CALL_KW whose
# `null_or_self` receiver slot (arg index 1) is the PY_NULL sentinel
# (GcRef(0)) for a plain no-receiver call.  The walker's residual-executor
# NULL-Ref-arg refusal used to lack the `is_call_kw` receiver-slot exemption
# that its sibling walker_abort_if_mayforce_null_ref_arg has, so it declined
# the recording iteration's call to a symbolic op and dropped that one
# iteration's effect — the hot-loop sum came out exactly one term short.
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
