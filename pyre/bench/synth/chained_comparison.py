N = 1400000


def main():
    acc = 0
    i = 0
    while i < N:
        # `0 < i < N` chained comparison lowers to SWAP + COPY (keeping the
        # middle operand on the stack across the short-circuit branch). SWAP
        # is an unported opcode that emits `abort_permanent` in the loop body,
        # which the full-body walk must decline up front — otherwise the walk
        # mis-seeds the loop guard, exits the loop early, and concretely
        # double-executes the tail (wrong result + doubled `print`).
        if 0 < i < N:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
