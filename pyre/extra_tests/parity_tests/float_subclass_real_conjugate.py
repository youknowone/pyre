import gc

# `descr_get_real` / `descr_conjugate` both return `space.float(self)`, so a
# subclass receiver is down-converted to a base `float` rather than handed back
# unchanged.  `imag` builds a fresh zero and was always exact.


class F(float):
    pass


f = F(1.25)
assert type(f.real) is float, type(f.real)
assert type(f.imag) is float, type(f.imag)
assert type(f.conjugate()) is float, type(f.conjugate())
assert f.real == 1.25
assert f.imag == 0.0
assert f.conjugate() == 1.25

# An exact float keeps its identity through all three.
g = 2.5
assert g.real is g
assert g.conjugate() is g

# complex keeps its own shape: real/imag are floats, conjugate stays complex.
class C(complex):
    pass


c = C(1, 2)
assert type(c.real) is float
assert type(c.imag) is float
assert type(c.conjugate()) is complex
assert c.conjugate() == complex(1, -2)

# int.conjugate has the same down-conversion contract.
class I(int):
    pass


i = I(7)
assert type(i.conjugate()) is int, type(i.conjugate())
assert type(i.real) is int, type(i.real)
assert type(i.imag) is int, type(i.imag)

# Growing the per-instance __slots__ storage allocates on every append, so the
# storage list can move mid-loop; a stale pointer would drop or misplace the
# writes.  Populate the highest slot first so the grow path runs, and force
# collections while it does.
def slot_grow(base, make):
    names = tuple(f"s{n}" for n in range(64))
    sub = type("Slotted", (base,), {"__slots__": names})
    obj = make(sub)
    for index in reversed(range(64)):
        setattr(obj, names[index], [index] * 8)
        gc.collect()
    return [getattr(obj, name)[0] for name in names]


assert slot_grow(str, lambda t: t("payload")) == list(range(64))
assert slot_grow(float, lambda t: t(1.5)) == list(range(64))
assert slot_grow(complex, lambda t: t(1, 2)) == list(range(64))

print("OK")
