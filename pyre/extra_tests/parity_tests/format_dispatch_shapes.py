# CPython-suite gap: `test_format` exercises the spec grammar and
# `test_types` the `__format__` protocol, but neither pins WHICH body a
# `format()` runs -- so a runtime is free to answer an exact `int` from its own
# formatter or by dispatching through `int.__format__` and nothing notices.
# parity-tests reason: pyre recognises the shared builtin `__format__` and
# formats the value directly instead of walking the generic call path. That is
# only sound while the two are the same function for every receiver, so this
# pins the answer for each shape the recognition has to get right: the builtins
# that share the body, `object.__format__` which does not, builtin subclasses
# with and without an override, and the descriptor spellings an override can
# take.

import decimal
import enum
import fractions
import re

results = []
ADDRESS = re.compile(r"object at 0x[0-9a-fA-F]+")


def row(label, fn):
    try:
        out = repr(fn())
    except BaseException as exc:  # noqa: BLE001 - the message is the assertion
        out = f"{type(exc).__name__}: {exc}"
    # A default `__repr__` prints an address; the identity of the object is not
    # what any of these rows is about.  Anchored on `object at ` so it cannot
    # reach a row whose subject IS hex -- `format(12345, '#x')`.  Assembled
    # outside the f-string: an expression part may not hold a backslash before
    # 3.12, and pypy3 is 3.11.
    out = ADDRESS.sub("object at 0xADDR", out)
    results.append(label + " -> " + out)


# The four builtins that publish the shared body, across the spec grammar.
for spec in ["", ">5", "<5", "^5", "05", "+d", " d", "-d", "#x", "#o", "#b",
             "x", "X", "o", "b", "c", "n", ",", "_", ",d", "_x", "e", "f",
             "g", "%", ".3f", "10.3f", "*>8", "é>6", "\ud800>6"]:
    row(f"int {spec!r}", lambda s=spec: format(12345, s))
    row(f"fstring int {spec!r}", lambda s=spec: f"{12345:{s}}")
for spec in ["", ">8", "05", ".3f", "e", "g", "%", "+.2f", "z.1f", "n"]:
    row(f"float {spec!r}", lambda s=spec: format(-0.0, s))
    row(f"float3.14 {spec!r}", lambda s=spec: format(3.14159, s))
for spec in ["", ">8", "<8", "^8", ".3", "8.3", "*^9", "s"]:
    row(f"str {spec!r}", lambda s=spec: format("abc", s))
for spec in ["", ">8", "d", "x", "05"]:
    row(f"bool {spec!r}", lambda s=spec: format(True, s))

# `object.__format__` carries its own body, so it must NOT be recognised: an
# empty spec renders `str(self)` and a non-empty one raises.
class Plain:
    pass


row("object ''", lambda: format(Plain(), ""))
row("object '>5'", lambda: format(Plain(), ">5"))


# A user `__format__`, and the descriptor spellings one can take.
class Override:
    def __format__(self, spec):
        return "OV:" + spec


class Static:
    @staticmethod
    def __format__(spec):
        return "SM:" + str(spec)


class Bound:
    def __get__(self, obj, objtype=None):
        return lambda spec: "DESC:" + spec


class WithDescriptor:
    __format__ = Bound()


row("override ''", lambda: format(Override(), ""))
row("override '>5'", lambda: format(Override(), ">5"))
row("staticmethod", lambda: format(Static(), "q"))
row("descriptor", lambda: format(WithDescriptor(), "q"))

# `__format__` is a type-level special method: an instance-dict entry is not it.
shadowed = Plain()
shadowed.__format__ = lambda spec: "SHADOW"
row("instance-dict shadow", lambda: format(shadowed, ""))


# Builtin subclasses: the inherited shared body, and one that overrides it.
class MyInt(int):
    pass


class MyStr(str):
    pass


class MyFloat(float):
    pass


class OverInt(int):
    def __format__(self, spec):
        return "OI:" + spec


row("MyInt ''", lambda: format(MyInt(42), ""))
row("MyInt '>6'", lambda: format(MyInt(42), ">6"))
row("MyStr ''", lambda: format(MyStr("hi"), ""))
row("MyStr '>6'", lambda: format(MyStr("hi"), ">6"))
row("MyFloat '.2f'", lambda: format(MyFloat(1.5), ".2f"))
row("OverInt '>6'", lambda: format(OverInt(42), ">6"))


# Types with a `__format__` of their own: recognising the shared body must not
# capture these.
class Color(enum.Enum):
    RED = 1


class IColor(enum.IntEnum):
    RED = 1


row("Decimal ''", lambda: format(decimal.Decimal("1.005"), ""))
row("Decimal '.2f'", lambda: format(decimal.Decimal("1.005"), ".2f"))
row("Decimal '>10'", lambda: format(decimal.Decimal("1.005"), ">10"))
row("Fraction ''", lambda: format(fractions.Fraction(1, 3), ""))
row("complex ''", lambda: format(1 + 2j, ""))
row("complex '>10'", lambda: format(1 + 2j, ">10"))
row("complex '.2f'", lambda: format(1 + 2j, ".2f"))
row("Enum ''", lambda: format(Color.RED, ""))
row("Enum '>10'", lambda: format(Color.RED, ">10"))
row("IntEnum ''", lambda: format(IColor.RED, ""))
row("IntEnum 'd'", lambda: format(IColor.RED, "d"))
row("IntEnum '>5'", lambda: format(IColor.RED, ">5"))

# Rejected specs still have to come from the right body.
row("int 'Q'", lambda: format(1, "Q"))
row("str 'd'", lambda: format("a", "d"))
row("int 's'", lambda: format(1, "s"))

for line in results:
    print(line)
print("OK")
