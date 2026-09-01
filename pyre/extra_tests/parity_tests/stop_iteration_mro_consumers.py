# CPython-suite gap: iterator consumers do not hot-loop mixed-MRO StopIteration subclasses.
# parity-tests reason: pins exhaustion classification without inflating the synth protocol gate.

"""Iterator consumers recognize StopIteration through either MRO order."""

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

print("OK")
