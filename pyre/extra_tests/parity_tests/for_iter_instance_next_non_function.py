# CPython-suite gap: the suite does not exercise non-function __next__
# descriptors at a hot FOR_ITER site.
# parity-tests reason: the user-function inline route must decline classmethod,
# builtin, and callable-instance forms without changing iterator semantics.


class ClassMethodIterator:
    remaining = 0

    def __init__(self, count):
        type(self).remaining = count

    def __iter__(self):
        return self

    @classmethod
    def __next__(cls):
        if cls.remaining == 0:
            raise StopIteration
        cls.remaining -= 1
        return cls.remaining


def builtin_iterator(count):
    source = iter(range(count))

    class BuiltinIterator:
        def __iter__(self):
            return self

        __next__ = source.__next__

    return BuiltinIterator()


class NextCallable:
    def __init__(self):
        self.remaining = 0

    def __call__(self):
        if self.remaining == 0:
            raise StopIteration
        self.remaining -= 1
        return self.remaining


class CallableIterator:
    __next__ = NextCallable()

    def __init__(self, count):
        type(self).__next__.remaining = count

    def __iter__(self):
        return self


def consume(iterator):
    count = 0
    for _ in iterator:
        count += 1
        if count == 1600:
            break
    return count


for _ in range(12):
    assert consume(ClassMethodIterator(1600)) == 1600
    assert consume(builtin_iterator(1600)) == 1600
    assert consume(CallableIterator(1600)) == 1600
print("OK")
