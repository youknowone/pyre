# list.__reversed__() exposed as a method (listobject.py:737 descr_reversed).

it = [1, 2, 3].__reversed__()
assert list(it) == [3, 2, 1]

# matches the reversed() builtin
assert list([1, 2, 3].__reversed__()) == list(reversed([1, 2, 3]))

# empty list
assert list([].__reversed__()) == []

# the method is iterable and exhausts once
it2 = ["a", "b"].__reversed__()
assert next(it2) == "b"
assert next(it2) == "a"
try:
    next(it2)
    assert False
except StopIteration:
    pass

print("list_reversed_dunder ok")
