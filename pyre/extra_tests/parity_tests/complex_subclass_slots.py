import gc


class Payload:
    pass


class Number(complex):
    __slots__ = ("prec",)

    def __new__(cls, *args, **kwargs):
        result = complex.__new__(cls, *args)
        result.prec = kwargs.get("prec", 12)
        return result


x = Number(3.14, prec=6)
assert x == 3.14
assert x.prec == 6
assert not hasattr(x, "__dict__")

x.prec = 7
assert x.prec == 7


def hot_slot_roundtrip(value):
    total = 0
    for i in range(2_000):
        value.prec = i
        total += value.prec
    return total


assert hot_slot_roundtrip(x) == 1_999 * 2_000 // 2

payload = Payload()
x.prec = payload
gc.collect()
assert x.prec is payload

del x.prec
try:
    x.prec
except AttributeError:
    pass
else:
    raise AssertionError("deleted slot remained bound")

try:
    del x.prec
except AttributeError:
    pass
else:
    raise AssertionError("deleting an unbound slot succeeded")
