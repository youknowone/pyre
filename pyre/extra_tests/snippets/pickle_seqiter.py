it = iter([1, 2, 3])
assert it.__reduce__() == (iter, ([1, 2, 3],), 0), it.__reduce__()
assert it.__length_hint__() == 3
next(it)
next(it)
r = it.__reduce__()
assert r == (iter, ([1, 2, 3],), 2), r
assert it.__length_hint__() == 1
it2 = r[0](*r[1])
it2.__setstate__(r[2])
assert list(it2) == [3]
assert iter((1, 2, 3)).__reduce__() == (iter, ((1, 2, 3),), 0)
assert iter("abc").__reduce__() == (iter, ("abc",), 0)
it3 = iter([9, 8])
it3.__setstate__(-5)
assert list(it3) == [9, 8]
print("pickle_seqiter OK")
