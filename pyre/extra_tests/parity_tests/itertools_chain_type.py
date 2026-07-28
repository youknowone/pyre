import gc
import itertools


assert isinstance(itertools.chain, type)
assert list(itertools.chain([1, 2], (), [3])) == [1, 2, 3]
assert list(itertools.chain.from_iterable([[1], [], [2, 3]])) == [1, 2, 3]

alias = itertools.chain[int]
assert alias.__origin__ is itertools.chain
assert alias.__args__ == (int,)

finalized = []


def make_subtype():
    def finalize(self):
        finalized.append("chain")

    return type("FinalizingChain", (itertools.chain,), {"__del__": finalize})


subtype = make_subtype()
direct = subtype([1])
alternate = subtype.from_iterable([[2]])
assert type(direct) is subtype
assert type(alternate) is subtype
assert list(direct) == [1]
assert list(alternate) == [2]
del direct, alternate, subtype
gc.collect()
assert finalized == ["chain", "chain"]

try:
    itertools.chain(source=[])
except TypeError:
    pass
else:
    raise AssertionError("chain accepted a keyword argument")


class KeywordChain(itertools.chain):
    def __init__(self, *iterables, marker):
        self.marker = marker


obj = KeywordChain([], marker=42)
assert obj.marker == 42

print("OK")
