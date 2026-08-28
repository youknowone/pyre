# CPython-suite gap: the suite does not exercise every iterator consumer with
# a multiply inherited StopIteration subclass in both base orders.
# parity-tests reason: iterator protocol exhaustion must use the exception MRO,
# not the first base's flattened exception tag.

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


class SV(StopIteration, ValueError):
    pass


class VS(ValueError, StopIteration):
    pass


class It:
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
    for value in It(3, stop_type):
        result.append(value)
    return result


def f(*args):
    return args


def exercise(stop_type):
    assert consume_for(stop_type) == [2, 1, 0]
    assert list(It(3, stop_type)) == [2, 1, 0]
    assert tuple(It(3, stop_type)) == (2, 1, 0)
    assert sum(It(3, stop_type)) == 3
    assert max(It(3, stop_type)) == 2
    assert next(It(0, stop_type), "dflt") == "dflt"
    assert f(*It(3, stop_type)) == (2, 1, 0)


for _ in range(3000):
    exercise(SV)
    exercise(VS)

print("OK")
