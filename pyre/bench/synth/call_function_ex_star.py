N = 300000


def add(a, b, c):
    return a + b + c


def helper(i):
    # `add(*args)` compiles to CALL_FUNCTION_EX.  Inside a callee that is
    # inlined into the hot loop, the unported opcode used to emit
    # abort_permanent and decline the callee's jitcode; the residual port
    # lets it compile.  Guards CALL_FUNCTION_EX output correctness and the
    # demonstrable inline (loops_aborted drops to 0).
    args = (i, 1, 2)
    return add(*args)


def main():
    total = 0
    for i in range(N):
        total += helper(i)
    print(total)


main()
