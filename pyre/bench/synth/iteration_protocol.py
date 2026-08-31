# pyre-check: max-pypy-ratio=173
N = 20000


class CountDown:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n = self.n - 1
        return self.n


def main():
    i = 0
    acc = 0
    while i < N:
        for x in CountDown(5):
            acc = acc + x
        i = i + 1
    print(acc)


main()


# Exhaustion is an MRO question even when StopIteration is not the first base.
# Keep it beside the hot iterator protocol it exercises instead of in a second
# parity driver.
class SV(StopIteration, ValueError):
    pass


class VS(ValueError, StopIteration):
    pass


class MixedStop:
    def __init__(self, n, stop_type):
        self.n = n
        self.stop_type = stop_type

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 0:
            raise self.stop_type("done")
        self.n -= 1
        return self.n


def consume_for(stop_type):
    result = []
    for value in MixedStop(3, stop_type):
        result.append(value)
    return result


def collect_args(*args):
    return args


for _ in range(3000):
    for stop_type in (SV, VS):
        assert list(MixedStop(3, stop_type)) == [2, 1, 0]
        assert tuple(MixedStop(3, stop_type)) == (2, 1, 0)
        assert sum(MixedStop(3, stop_type)) == 3
        assert next(MixedStop(0, stop_type), "done") == "done"
        assert consume_for(stop_type) == [2, 1, 0]
        assert max(MixedStop(3, stop_type)) == 2
        assert collect_args(*MixedStop(3, stop_type)) == (2, 1, 0)
