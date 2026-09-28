# CPython-suite gap: no test stores __slots__ members on float/complex/str/bytearray subclasses under a hot loop.
# parity-tests reason: this targets the typedef.py _getusercls mapdict storage behind slot members.

"""`__slots__` members of float/complex/str/bytearray subclasses.

A slots-only subclass has no `__dict__`; a subclass that also asks for
`__dict__` must still keep the slot value in the slot, not in the dict.
"""

# int, tuple and bytes reject a nonempty __slots__.
BASES = [
    (float, (1.5,)),
    (complex, (1 + 2j,)),
    (str, ("ab",)),
    (bytearray, (b"ab",)),
]

N = 2000


def check(base, args):
    SlotsOnly = type("S" + base.__name__, (base,), {"__slots__": ("x",)})
    WithDict = type("D" + base.__name__, (base,), {"__slots__": ("x", "__dict__")})

    total = 0
    for i in range(N):
        s = SlotsOnly(*args)
        s.x = i
        total += s.x
        del s.x
        try:
            s.x
        except AttributeError:
            pass
        else:
            raise AssertionError("deleted slot still readable on %s" % base.__name__)
        try:
            s.y = 1
        except AttributeError:
            pass
        else:
            raise AssertionError("slots-only %s accepted a new attribute" % base.__name__)
        assert not hasattr(s, "__dict__"), base

        d = WithDict(*args)
        d.x = i
        d.y = i
        assert "x" not in d.__dict__, (base, d.__dict__)
        assert d.__dict__ == {"y": i}, (base, d.__dict__)
        total += d.x + d.y
    assert total == 3 * N * (N - 1) // 2, (base, total)


for base, args in BASES:
    check(base, args)
print("OK")
