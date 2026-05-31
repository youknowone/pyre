# Native iterators expose __next__ and __iter__ (next()/for already work;
# this covers the explicit dunder calls).

it = iter([1, 2])
assert it.__iter__() is it
assert it.__next__() == 1
assert it.__next__() == 2
try:
    it.__next__()
    raise AssertionError("expected StopIteration")
except StopIteration:
    pass

# Sequence-iterator family (all share the seq-iter type).
assert iter((1, 2)).__next__() == 1
assert iter("ab").__next__() == "a"
assert iter({1, 2}).__next__() in (1, 2)
assert iter(b"ab").__next__() == 97
assert reversed([10, 20]).__next__() == 20
assert zip([1], [2]).__next__() == (1, 2)
assert map(str, [1]).__next__() == "1"

# Distinct iterator types.
assert iter(range(3)).__next__() == 0
assert iter({1: 2}).__next__() == 1
assert iter({1: 2}.values()).__next__() == 2
assert iter({1: 2}.items()).__next__() == (1, 2)

en = enumerate("xy", 1)
assert en.__iter__() is en
assert en.__next__() == (1, "x")

# Generators and itertools count/repeat: __iter__() returns self.
def gen():
    yield 1
    yield 2
g = gen()
assert g.__iter__() is g
assert g.__next__() == 1

import itertools
c = itertools.count(5)
assert c.__iter__() is c
assert c.__next__() == 5
r = itertools.repeat(7, 2)
assert r.__iter__() is r
assert r.__next__() == 7

print("iterator_dunders ok")
