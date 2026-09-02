# A CALL_KW the seeded inline cannot take, so the call reaches the residual
# executor.  `**kw` is the shape that keeps it there: the permutation that
# seeds a keyword call's parameters (fbw_reorder_call_kw_args) has no slot for
# a surplus mapping, so the inline declines and the call stays a residual
# CallMayForce.
#
# That residual is what the NULL-Ref-arg refusal sees, and its arg index 1 is
# the `null_or_self` receiver slot, holding the PY_NULL sentinel (GcRef(0)) of
# a plain no-receiver call.  The refusal carries an `is_call_kw` exemption for
# it because that null is a checked sentinel, not an unresolved ref.  Without
# the exemption the refusal declines the recording iteration's call to a
# symbolic op and drops that iteration's effect, leaving the sum exactly one
# term short.
#
# call_kw_hot_loop.py is the other half: the same call spelling with a callee
# the inline does take, so the two cover the inlined and residual routes of
# one opcode.  The result depends on both the positional and the keyword arg
# so the call cannot be constant-folded away, and the exact aggregate makes a
# single dropped iteration observable.
N = 200000


def g(x, **kw):
    return x + kw["step"]


def main():
    total = 0
    for i in range(N):
        total += g(i, step=2)
    print(total)


main()
