# CPython-suite gap: exception tests omit this PyPy instance-dict storage path.
# parity-tests reason: it guards PyPy/pyre exception mapdict behavior.

"""A user attribute on an exception instance lives in its instance dict, which
is mapdict-backed.  Each block below runs the read hot enough to compile, then
perturbs the one thing that has to invalidate the compiled read.
"""

N = 3000


class E(Exception):
    pass


def read_sum(e):
    n = 0
    for _ in range(N):
        n += e.tag
    return n


def read_last(e):
    out = None
    for _ in range(N):
        out = e.tag
    return out


# An unboxed int slot, rewritten in place, then retyped so the slot migrates
# off unboxed storage.
e = E()
e.tag = 7
assert read_sum(e) == 7 * N
e.tag = 9
assert read_sum(e) == 9 * N
e.tag = "s"
assert read_last(e) == "s"

# A boxed slot, then a second attribute grows the map.
boxed = E()
boxed.tag = "seven"
assert read_last(boxed) == "seven"
boxed.other = 1
assert read_last(boxed) == "seven"

# Deleting the attribute.
gone = E()
gone.tag = 11
read_sum(gone)
del gone.tag
try:
    gone.tag
except AttributeError:
    pass
else:
    raise AssertionError("deleted attribute still readable")

# A second instance whose map places the attribute at another index, read at
# the same site.
shifted = E()
shifted.pad = 0
shifted.tag = 13
assert read_sum(shifted) == 13 * N
assert read_last(boxed) == "seven"

# Devolving the dictionary out from under the read.
devolved = E()
devolved.tag = 21
read_sum(devolved)
d = devolved.__dict__
for i in range(200):
    d["k%d" % i] = i
assert read_sum(devolved) == 21 * N
assert devolved.__dict__["tag"] == 21

# Writing through __dict__ rather than through the attribute.
written = E()
written.tag = 31
read_sum(written)
written.__dict__["tag"] = 32
assert read_sum(written) == 32 * N


# A class attribute of the same name: the instance attribute wins until it is
# deleted.
class F(E):
    pass


f = F()
f.tag = 41
assert read_sum(f) == 41 * N
F.tag = 99
assert read_sum(f) == 41 * N
del f.tag
assert read_sum(f) == 99 * N

# A data descriptor added to the class shadows the instance dictionary.
shadowed = E()
shadowed.tag = 3
assert read_sum(shadowed) == 3 * N
E.tag = property(lambda self: 77)
assert read_sum(shadowed) == 77 * N
del E.tag


# A non-exception receiver reaching the same read site.
class O:
    pass


o = O()
o.tag = 5
assert read_sum(o) == 5 * N

# The typed `args` slot is not a dictionary attribute.
assert E(1, 2).args == (1, 2)

print("OK")
