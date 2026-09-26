# CPython-suite gap: no suite test calls `len()` on a user `__len__` whose
# answer changes, including a negative one, in a loop hot enough to compile
# a bridge through the operator.
#
# parity-tests reason: the bridge walks `len`'s own tail after `__len__`
# returns. A result baked at trace time, or a tail that skips the length
# check, prints a different total and error count.
#
# A hot `len()` whose `__len__` answers a different value on every call,
# including a negative that raises ValueError. The bridge through the
# operator must re-run the length check; a constant baked at trace time
# would keep answering the first call.


class Left:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class Right:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def main():
    left = Left(3)
    right = Right(8)
    neg = Left(-1)
    total = 0
    errors = 0
    seen = []
    # Stay on Left long enough to trace, then switch. A bridge that baked
    # the first answer would keep returning 3.
    for i in range(8000):
        if i < 4000:
            obj = left
        elif i % 17 == 0:
            obj = neg
        else:
            obj = right
        try:
            n = len(obj)
        except ValueError:
            errors += 1
            n = -1
        total += n
        if i in (0, 3999, 4000, 4001, 4017, 7999):
            seen.append((i, n))
    print(total, errors, seen)


main()
