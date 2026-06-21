# UNPACK_EX in a hot loop: `a, *b = data` with a star target compiles to the
# unpack_ex residual (the UNPACK_SEQUENCE sibling with a starred list slot)
# instead of an abort_permanent marker, so the hot body JIT-compiles rather
# than declining to the trait leg. The residual returns a tuple of the
# `before + 1 + after` slots that `unpack_item_fn` reads back out.
# Output is verified against CPython/PyPy.
N = 200000


def main():
    total = 0
    i = 0
    data = [1, 2, 3, 4]
    while i < N:
        a, *b = data
        total = total + a + b[0] + b[1] + b[2]
        i = i + 1
    print(total)


main()
