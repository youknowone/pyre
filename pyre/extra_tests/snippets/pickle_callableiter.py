import pickle


class Counter:
    def __init__(self):
        self.value = 0

    def __call__(self):
        value = self.value
        self.value += 1
        return value


callable_iterator = iter(Counter(), 3)
reconstructor, args = callable_iterator.__reduce__()
assert reconstructor is iter
assert args[1] == 3

for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    restored = pickle.loads(pickle.dumps(callable_iterator, protocol))
    assert list(restored) == [0, 1, 2]

exhausted = iter(lambda: 1, 1)
assert list(exhausted) == []
assert exhausted.__reduce__() == (iter, ((),))
