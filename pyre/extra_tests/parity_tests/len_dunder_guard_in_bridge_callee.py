# CPython-suite gap: no suite test fails a guard inside a `__len__` that a
# compiled bridge resumed into, with the answer going negative, float or bool.
#
# parity-tests reason: the resumed frame chain has to keep `len`'s tail
# between the caller and `__len__`; resuming straight to the caller returns
# the raw dunder result.


class Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        n = self.n
        if n > 1000:
            return -n
        if n == 500:
            return 2.5
        if n == 250:
            return True
        return n


def main():
    objs = [Sized(i % 64) for i in range(64)]
    total = 0
    counts = {}
    for i in range(12000):
        obj = objs[i % 64]
        if i > 6000 and i % 97 == 0:
            obj = Sized(2000)
        elif i > 7000 and i % 89 == 0:
            obj = Sized(500)
        elif i > 8000 and i % 83 == 0:
            obj = Sized(250)
        try:
            total += len(obj)
        except (ValueError, TypeError) as e:
            key = type(e).__name__
            counts[key] = counts.get(key, 0) + 1
    print(total, sorted(counts.items()))


main()
