# Loop-carried short-circuit `or` exercised on the FALSY leg, where the loop
# lives in a callee invoked repeatedly from a hot outer loop.
#
# `x = x or b` lowers to a `POP_JUMP_IF_TRUE` over `POP_TOP; LOAD b; STORE x`.
# When the callee `f` is hot enough to compile as a function-entry trace, the
# next call re-enters the compiled loop through the function-entry path.  With
# `x` falsy (`None`), the truthy-assuming guard fails; the blackhole resumes
# the not-taken arm and reaches the loop header, raising ContinueRunningNormally
# with the merge-point PC in its green args.  The function-entry resume must
# write that merge-point PC back to the frame before re-entering the
# interpreter.  Skipping it leaves the frame at the guard PC (operand depth 2)
# while the merge-point restore set depth 0, so the next pop underflows
# ("stack underflow during interpreter opcode").  The loop-back-edge resume
# already writes the PC; the function-entry resume must match it.
#
# Deterministic: `s` counts the calls that returned the kept `b`.
N = 4000


def f(a, b, n):
    x = a
    i = 0
    while i < n:
        x = x or b
        i = i + 1
    return x


def run():
    s = 0
    i = 0
    while i < N:
        if f(None, 5, 50) == 5:
            s = s + 1
        i = i + 1
    return s


def main():
    print(run())


main()
