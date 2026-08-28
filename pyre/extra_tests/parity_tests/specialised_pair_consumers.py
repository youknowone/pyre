# CPython-suite gap: tuple consumers do not cover PyPy specialised-pair layouts.
# parity-tests reason: this targets PyPy/pyre pair storage, JIT guards, and consumers.

"""Index, unpack, compare, and materialize every specialised pair shape."""

BIG = (1 << 4095) + (1 << 2001) + 0x123456789
N = 3000


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class Pair(tuple):
    def __len__(self):
        return 99

    def __getitem__(self, index):
        return 7


def produced_pairs():
    """One consumer site alternates ii, oo-long, oo-float, and mixed pairs."""
    checksum = 0
    for i in range(N):
        kind = i & 3
        if kind == 0:
            pair = divmod(i + 7919, 97)
        elif kind == 1:
            pair = divmod(BIG + i, (1 << 755) + 0xABCDEF)
        elif kind == 2:
            pair = (i + 0.5, i + 1.5)
        else:
            pair = (i, "x")
        first, second = pair
        assert len(pair) == 2
        assert pair[0] is first and pair[-1] is second
        checksum ^= int(first) & 0xFFFF
        checksum ^= (int(second) if not isinstance(second, str) else len(second)) & 0xFFFF
    return checksum


def index_guards():
    checksum = 0
    keys = (0, 1, -1, -2, True, False, Index(0), Index(1))
    for i in range(N):
        pair = (i, i * 7 + 3)
        checksum ^= pair[keys[i & 7]]
        assert pair[:] == pair and pair[0:1] == (i,)
        try:
            pair[2]
        except IndexError:
            pass
        else:
            raise AssertionError("pair[2] did not raise")
    return checksum


def other_consumers():
    ii = (11, 22)
    oo = ("ab", "cd")
    ff = (1.5, 2.5)
    nested = ((1, 2), (3, 4))
    for _ in range(N):
        assert ii == (11, 22) and ii < (11, 23)
        assert 11 in ii and 33 not in ii and "ab" in oo
        assert ff <= (1.5, 2.5)
        a, b = nested
        assert a[1] == 2 and b[0] == 3
        odd = Pair(ii)
        assert len(odd) == 99 and odd[0] == 7


def identity():
    x = float("0.125")
    y = float("0.25")
    same = (x, x)
    distinct = (x, y)
    assert same[0] is x and same[1] is x and same[0] is same[1]
    assert distinct[0] is x and distinct[1] is y and distinct[0] is not distinct[1]


produced_pairs()
index_guards()
other_consumers()
identity()
print("OK")
