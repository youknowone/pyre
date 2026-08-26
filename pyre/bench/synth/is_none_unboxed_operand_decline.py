# `optional_none_arg_inline` with the default changed from `None` to an int, so
# the local the `is`-against-None branch tests is the one the callee's register
# banks hold unboxed.  That is the case the scan exists for: a mid-body guard
# resume cannot source the operand's Ref form from those banks, the encoded
# liveness stream disagrees with the decoder and the caller frame is corrupted.
# `_read_from_buffer(self, size=-1)` was the shape that miscompiled.
#
# The gate is `loops_compiled` reading 2 -- the callee keeps a trace of its own.
# Dropping the scan folds it into the caller's and the count falls to 1, which
# is the movement this fixture is here to catch.
N = 200000


def read_n(buf, size=-1):
    if size is None or size < 0:
        return buf
    return size


def main():
    total = 0
    for i in range(N):
        total += read_n(i)
    total += read_n(1, 7)
    print(total)


main()
