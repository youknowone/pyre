"""Three entries that accepted a surplus argument instead of counting it.

`typeobject.py:1028-1031 descr__init__` reads nothing — `__new__` already built
the class — so all it does is settle the count, over the arguments behind the
receiver only.  `eval` and `exec` bind `globals` / `locals` positionally and
have nowhere to put a fourth.  A sequence repeat takes its count through
`getindex_w`, which is where a count too large for a machine word is caught, so
no receiver may answer before that conversion runs.
"""

import sys

TYPE_INIT_COUNT = "type.__init__() takes 1 or 3 arguments"


def message(fn):
    try:
        result = fn()
    except TypeError as exc:
        return str(exc)
    raise AssertionError(f"expected TypeError, got {result!r}")


def overflows(label, fn):
    try:
        result = fn()
    except OverflowError as exc:
        assert str(exc) == "cannot fit 'int' into an index-sized integer", (label, str(exc))
        return
    raise AssertionError(f"{label} returned {result!r} instead of raising OverflowError")


# Only one or three arguments behind the receiver; keywords are not counted, so
# a keywords-only call is refused for having none.
assert message(lambda: type.__init__(int)) == TYPE_INIT_COUNT
assert message(lambda: type.__init__(int, 1, 2)) == TYPE_INIT_COUNT
assert message(lambda: type.__init__(int, 1, 2, 3, 4)) == TYPE_INIT_COUNT
assert message(lambda: type.__init__(int, x=1)) == TYPE_INIT_COUNT

assert type.__init__(int, 1) is None
assert type.__init__(int, "A", (), {}) is None
assert type.__init__(int, "A", (), {}, x=1) is None


# The counts the interpreter itself goes through when it builds a class.
class Meta(type):
    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace)


class Built(metaclass=Meta):
    pass


assert type(Built) is Meta
assert type("Plain", (), {"x": 1}).x == 1
assert type(1) is int

# `eval` and `exec` have three positional slots; `exec`'s `closure` is
# keyword-only, so a fourth positional is refused too.
for label, fn in [
    ("eval 4", lambda: eval("1", {}, {}, 1)),
    ("eval 5", lambda: eval("1", {}, {}, 1, 2)),
    ("exec 4", lambda: exec("1", {}, {}, 1)),
    ("exec 5", lambda: exec("1", {}, {}, 1, 2)),
]:
    message(fn)

assert eval("1", {}, {}) == 1
assert exec("x = 1", {}, {}) is None

if sys.implementation.name != "pypy":
    assert message(lambda: eval("1", {}, {}, 1)) == "eval() takes at most 3 arguments (4 given)"
    assert message(lambda: eval("1", {}, {}, 1, 2)) == "eval() takes at most 3 arguments (5 given)"
    assert (
        message(lambda: exec("1", {}, {}, 1))
        == "exec() takes at most 3 positional arguments (4 given)"
    )
    assert message(lambda: exec("1", {}, {}, 1, 2)) == "exec() takes at most 4 arguments (5 given)"

# Both operands are literals here, so the bytecode constant folder sees the
# repeat before the interpreter does.  It has to decline when the count does not
# convert, exactly as the runtime conversion below would refuse it.
for label, fn in [
    ("(1,) * 2**63", lambda: (1,) * (2**63)),
    ("[] * 2**63", lambda: [] * (2**63)),
    ("[1] * 2**63", lambda: [1] * (2**63)),
    ("'' * 2**63", lambda: "" * (2**63)),
    ("b'' * 2**63", lambda: b"" * (2**63)),
    ("bytearray() * 2**63", lambda: bytearray() * (2**63)),
]:
    overflows(label, fn)

if sys.implementation.name != "pyre":
    # An empty tuple times an unconvertible count is the one repeat the folder
    # answers instead of declining: `const_folding_safe_multiply` returns the
    # empty result from `elements.is_empty()` before it converts the count, so
    # the expression never reaches the interpreter.  The str and bytes arms
    # convert first, which is why they are asserted above.  The folder lives in
    # the pinned rustpython-codegen dependency, not in this tree; move these
    # three rows up with the rest once that pin advances past the fix.
    for label, fn in [
        ("() * 2**63", lambda: () * (2**63)),
        ("() * 2**64", lambda: () * (2**64)),
        ("(2**63) * ()", lambda: (2**63) * ()),
    ]:
        overflows(label, fn)

# The same through runtime values and the explicit slot, which no fold sees.
HUGE = 2**63
EMPTY_TUPLE = ()
EMPTY_STR = ""
EMPTY_BYTES = b""
for label, fn in [
    ("t * n", lambda: EMPTY_TUPLE * HUGE),
    ("n * t", lambda: HUGE * EMPTY_TUPLE),
    ("t.__mul__(n)", lambda: EMPTY_TUPLE.__mul__(HUGE)),
    ("t.__rmul__(n)", lambda: EMPTY_TUPLE.__rmul__(HUGE)),
    ("tuple.__mul__(t, n)", lambda: tuple.__mul__(EMPTY_TUPLE, HUGE)),
    ("s * n", lambda: EMPTY_STR * HUGE),
    ("b * n", lambda: EMPTY_BYTES * HUGE),
    ("list * n", lambda: [] * HUGE),
]:
    overflows(label, fn)

# A count that does fit still repeats, and zero or negative empties.
assert () * 3 == ()
assert (1, 2) * 2 == (1, 2, 1, 2)
assert (1, 2) * 0 == ()
assert (1, 2) * -1 == ()
assert [] * 3 == []

print("OK")
