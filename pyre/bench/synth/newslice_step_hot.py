# BUILD_SLICE with a step operand records a `newslice(start, stop, step)` HLOp,
# lowered to a build_slice residual call.  Without that lowering arm the op
# reached the assembler un-lowered and panicked whenever a stepped slice (e.g.
# `lst[::-1]`) sat in a JIT-compiled loop.  The hot body below builds a
# literal-step, a variable-step, and a bounded negative-step slice each
# iteration so the residual JIT-compiles instead of declining.
# Output is verified against CPython/PyPy.
N = 200000


def main():
    lst = list(range(10))
    step = -1
    total = 0
    i = 0
    while i < N:
        rev = lst[::-1]
        var = lst[::step]
        bounded = lst[8:2:-1]
        total = total + rev[0] + var[1] + bounded[-1]
        i = i + 1
    print(total)


main()
