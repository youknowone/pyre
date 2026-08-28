# CPython-suite gap: the suite does not exhaust a hot user-defined __next__
# with a StopIteration subclass.
# parity-tests reason: FOR_ITER exhaustion matching is by subclass, including
# after the iterator method has entered the trace.


class MyStop(StopIteration):
    pass


class SubclassStoppingIterator:
    def __init__(self, limit):
        self.index = 0
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise MyStop
        self.index += 1
        return self.index


def consume(limit):
    count = 0
    for _ in SubclassStoppingIterator(limit):
        count += 1
    return count


for _ in range(12):
    assert consume(1600) == 1600
print("OK")
