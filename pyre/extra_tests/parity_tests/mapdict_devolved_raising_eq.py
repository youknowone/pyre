"""A devolved instance dictionary probed by attribute name.

The probe compares the name against whatever each colliding bucket holds, so a
stored non-string key can reach a user `__eq__`.  When that raises, the read
must propagate the exception; reporting it as a miss would silently let a class
attribute of the same name answer instead.

The lone-surrogate blocks exercise the other half of the name dispatch: a name
that is not valid UTF-8 has no borrowed-str view, so it wraps before probing.
"""


class Colliding:
    """Hashes as `"zz"` and refuses to compare."""

    def __hash__(self):
        return hash("zz")

    def __eq__(self, other):
        raise ValueError("boom")


class Quiet:
    """Hashes as `"zz"` and compares unequal without raising."""

    def __hash__(self):
        return hash("zz")

    def __eq__(self, other):
        return NotImplemented


def devolve(obj):
    """Grow the instance dict past the mapdict limit, then return it."""
    for i in range(200):
        setattr(obj, "a%d" % i, i)
    return obj.__dict__


class R:
    zz = "CLASSVALUE"


# A raising comparison in the probe surfaces, and the class attribute does not
# win by default.
r = R()
devolve(r)[Colliding()] = 1
try:
    r.zz
except ValueError as exc:
    assert str(exc) == "boom", str(exc)
else:
    raise AssertionError("a raising __eq__ in the probe was reported as a miss")

# getattr and __getattribute__ reach the same probe.
for read in (lambda o: getattr(o, "zz"), lambda o: type(o).__getattribute__(o, "zz")):
    try:
        read(r)
    except ValueError:
        pass
    else:
        raise AssertionError("raising __eq__ swallowed on an alternate read path")

# The dict subscript itself agrees.
try:
    r.__dict__["zz"]
except ValueError:
    pass
except KeyError:
    raise AssertionError("raising __eq__ reported as a missing key")

# A colliding key that compares unequal without raising is an ordinary miss, so
# the class attribute answers.
q = R()
devolve(q)[Quiet()] = 1
assert q.zz == "CLASSVALUE"

# An instance attribute still wins over the class attribute after devolving.
own = R()
devolve(own)
own.zz = "OWN"
assert own.zz == "OWN"
own.__dict__[Quiet()] = 1
assert own.zz == "OWN"

# A builtin subclass reaches the same terminator.
import _random  # noqa: E402


class Rand(_random.Random):
    zz = "CLASSVALUE"


rand = Rand()
devolve(rand)[Colliding()] = 1
try:
    rand.zz
except ValueError:
    pass
else:
    raise AssertionError("raising __eq__ swallowed on a builtin subclass")

# A lone-surrogate attribute name takes the wrapping arm of the name dispatch.
SURROGATE = "z\udcffz"


class S:
    pass


s = S()
devolve(s)
setattr(s, SURROGATE, "SURR")
assert getattr(s, SURROGATE) == "SURR"
assert s.__dict__[SURROGATE] == "SURR"

# ... and it propagates a raising comparison too.  `Colliding` hashes as "zz",
# so pick a colliding key for this name instead.
class CollidingSurrogate:
    def __hash__(self):
        return hash(SURROGATE)

    def __eq__(self, other):
        raise ValueError("boom")


s2 = S()
devolve(s2)
s2.__dict__[CollidingSurrogate()] = 1
try:
    getattr(s2, SURROGATE)
except ValueError:
    pass
except AttributeError:
    raise AssertionError("raising __eq__ reported as a missing attribute")

print("OK")
