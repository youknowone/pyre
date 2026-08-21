# pyre-check: gate=1
"""A bound's own `__index__` may resize the sequence it bounds.

Converting the bounds is the step that runs Python, so the length that folds a
negative bound and clamps a slice has to be read after that step, not before.
"""


def grow(target, value):
    class Bound:
        def __index__(self):
            target.extend(range(100, 120))
            return value

    return Bound()


numbers = list(range(10))
assert numbers[grow(numbers, 8) :] == [8, 9] + list(range(100, 120)), numbers[8:]

numbers = list(range(10))
assert numbers[grow(numbers, -3) :] == [117, 118, 119]

numbers = list(range(10))
assert numbers[grow(numbers, -3) : -1] == [117, 118]

numbers = list(range(10))
del numbers[grow(numbers, 8) :]
assert numbers == list(range(8)), numbers

numbers = list(range(10))
numbers[grow(numbers, 8) :] = [1]
assert numbers == list(range(8)) + [1], numbers

# A negative `start` folds against the length the conversion left behind, so
# the value at index 15 is outside the window that starts at 25.
numbers = list(range(10))
try:
    found = numbers.index(105, grow(numbers, -5))
except ValueError:
    found = "not in list"
assert found == "not in list", found

numbers = list(range(10))
assert numbers.index(105, grow(numbers, 0)) == 15

buf = bytearray(b"abcdefghij")


def grow_buf(value):
    class Bound:
        def __index__(self):
            buf.extend(b"Z" * 20)
            return value

    return Bound()


assert buf.find(b"Z", grow_buf(-5)) == 25
