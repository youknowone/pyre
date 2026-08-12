"""CPython 3.14 identity guarantees for numeric objects stored in lists."""

import copy


def identity(value):
    return value


for value in (int("257"), float("1.5")):
    direct = [value]
    assert direct[0] is value
    assert direct[0] is direct[0]

    appended = []
    appended.append(value)
    assert appended[0] is value

    assert identity(value) is value


def hot_store_identity():
    ints = [int("257")]
    floats = [float("1.5")]
    i = 0
    while i < 5000:
        int_value = i + 257
        float_value = i + 0.25
        ints[0] = int_value
        floats[0] = float_value
        assert ints[0] is int_value
        assert floats[0] is float_value
        assert ints[0] is ints[0]
        assert floats[0] is floats[0]
        i += 1


hot_store_identity()


immutable_tuple = ((1, 2), 3)
assert copy.deepcopy(immutable_tuple) is immutable_tuple

memo = {}
copy.deepcopy([1, 2, 3, 4], memo)
assert len(memo) == 2, memo

memo = {}
copy.deepcopy([(1, 2)], memo)
assert len(memo) == 2, memo
