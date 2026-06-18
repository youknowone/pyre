assert list(zip(["a", "b", "c"], range(3), [9, 8, 7, 99])) == [
    ("a", 0, 9),
    ("b", 1, 8),
    ("c", 2, 7),
]

assert list(zip(["a", "b", "c"])) == [("a",), ("b",), ("c",)]
assert list(zip()) == []

assert list(zip(*zip(["a", "b", "c"], range(1, 4)))) == [("a", "b", "c"), (1, 2, 3)]


# test infinite iterator
class Counter(object):
    def __init__(self, counter=0):
        self.counter = counter

    def __next__(self):
        self.counter += 1
        return self.counter

    def __iter__(self):
        return self


it = zip(Counter(), Counter(3))
assert next(it) == (1, 4)
assert next(it) == (2, 5)


# `zip` is a lazy iterator (`W_Zip`); __reduce__ is (zip, (*iterators)).
z = zip([1, 2], [3, 4])
assert iter(z) is z
recon, args = z.__reduce__()
assert list(args[0]) == [1, 2] and list(args[1]) == [3, 4]

# The strict= flag round-trips through __reduce__ / __setstate__.
zs = zip([1, 2], [3, 4], strict=True)
assert zs.__reduce__()[2] is True
z = zip([1, 2], [3, 4])
assert len(z.__reduce__()) == 2
z.__setstate__(1)
assert z.__reduce__()[2] is True


# strict mismatch errors name the offending argument.
def expect_value_error(it, msg):
    try:
        list(it)
        raise AssertionError("no error")
    except ValueError as e:
        assert str(e) == msg, (str(e), msg)


expect_value_error(
    zip([1, 2, 3], [4, 5], strict=True), "zip() argument 2 is shorter than argument 1"
)
expect_value_error(
    zip([1, 2], [4, 5, 6], strict=True), "zip() argument 2 is longer than argument 1"
)
expect_value_error(
    zip([1, 2, 3], [4, 5, 6], [7, 8], strict=True),
    "zip() argument 3 is shorter than arguments 1-2",
)
