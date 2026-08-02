# Python 3.14's zip tuple retains the exact objects returned by each input
# iterator, including exact int and float objects that PyPy may store unboxed.


class Keeper:
    def __init__(self, value):
        self.value = value
        self.done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        self.done = True
        return self.value


def check(value):
    pair = next(zip(Keeper(value), Keeper(value)))
    assert pair[0] is value
    assert pair[1] is value
    assert pair[0] is pair[1]


check(int("100000000000000000000000000000000000000"))
check(float("0.125"))
print("OK")
