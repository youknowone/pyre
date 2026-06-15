# Exercises short-circuit `and` / `or` in VALUE context inside a hot loop.
#
# `x = (i % 7) and (i + 3)` lowers to `BINARY_OP %; COPY 1; TO_BOOL;
# POP_JUMP_IF_*`: the COPY-duplicated left operand stays on the operand
# stack across the branch guard, and the guard's NOT-taken arm resumes
# with that kept temp live (the #124 kept-stack snapshot shape).  Both the
# `and` (keeps the falsy left = 0) and `or` (keeps the truthy left) forms
# are covered, with a flip frequency (every 7th / every 3rd iteration) that
# makes the guard fail on a varying schedule.
#
# Pure arithmetic only, so the checksum is deterministic across runtimes —
# a parity regression target for the kept-stack branch-guard path.
N = 200000


def main():
    total = 0
    i = 0
    while i < N:
        a = (i % 7) and (i + 3)
        b = (i % 3) or (i + 7)
        c = (i % 5) and (i - 1) or (i + 2)
        total = total + a + b + c
        i = i + 1
    print(total)


main()
