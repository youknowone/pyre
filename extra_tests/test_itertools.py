import sys
import itertools

def test_chain():
    it = itertools.chain([], [1, 2, 3])
    lst = list(it)
    assert lst == [1, 2, 3]

def test_islice_maxint():
    slic = itertools.islice(itertools.count(), 1, 10, sys.maxsize)
    assert len(list(slic)) == 1

def test_islice_largeint():
    slic = itertools.islice(itertools.count(), 1, 10, sys.maxsize - 20)
    assert len(list(slic)) == 1


def test_generic_sequence_iterator_exhaustion_after_jit():
    class Sequence:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            return self.values[index]

    total = 0
    for _ in range(20):
        for value in Sequence(range(1000)):
            total += value
    assert total == 9_990_000
