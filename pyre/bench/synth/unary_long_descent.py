# pyre-check: spec-folds=unary_positive_descent,unary_invert_descent
# Keep both operands loop-carried and outside the machine-int range.  The fold
# layer never served these arms: the gate proves that the whole `pos` and
# `invert` interpreter bodies admit exact W_LongObject operands instead of
# merely preserving the former exact-int fast paths.
N = 200000
LONG = 1 << 70


def positive_long(n):
    x = LONG
    i = 0
    while i < n:
        x = +x
        i += 1
    return x


def invert_long(n):
    x = LONG
    i = 0
    while i < n:
        x = ~x
        i += 1
    return x


print(positive_long(N))
print(invert_long(N))
