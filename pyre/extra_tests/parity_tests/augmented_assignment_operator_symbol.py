# CPython-suite gap: test_augassign checks results, never the TypeError text,
# so nothing in the suite reads the symbol an augmented assignment reports.
# parity-tests reason: the message is identical on CPython 3.14 and PyPy
# 7.3.20, so all three runtimes are asserted against the same literal.

"""An augmented assignment reports its own symbol, not the operator's.

``_make_binop_impl`` and ``_make_inplace_impl`` generate two families from one
dispatch; the only thing separating them is the symbol baked into
``errormsg``.  Sharing the dispatch and dropping the symbol makes ``x &= y``
answer ``for &``, which points the reader at an operator the source does not
contain.

Only messages CPython and PyPy spell the same way are asserted here.  The
sequence concatenations (``b'a' += 1``, ``[1] += 1``) and the sequence repeat
(``1.5 *= 'a'``) diverge between the two oracles and so are left out.
"""


def message(code):
    try:
        exec(code, {})
    except TypeError as exc:
        return str(exc)
    raise AssertionError(f"{code!r} did not raise TypeError")


def expect(code, text):
    got = message(code)
    assert got == text, f"{code!r}\n  expected {text!r}\n  got      {got!r}"


# Every in-place numeric/bitwise operator names its augmented spelling.
for symbol in ["+", "&", "|", "^", "<<", ">>", "//", "%", "/", "**", "@"]:
    expect(
        f"x = 1\nx {symbol}= 'a'",
        f"unsupported operand type(s) for {symbol}=: 'int' and 'str'",
    )

# ... and the plain operator keeps naming itself.
for symbol in ["+", "&", "|", "^", "<<", ">>", "//", "%", "/", "@"]:
    expect(
        f"x = 1 {symbol} 'a'",
        f"unsupported operand type(s) for {symbol}: 'int' and 'str'",
    )
expect("x = 1 ** 'a'", "unsupported operand type(s) for ** or pow(): 'int' and 'str'")

# A left operand whose type declines through the generic path, rather than
# through a numeric fast path.
expect("x = []\nx -= 1", "unsupported operand type(s) for -=: 'list' and 'int'")
expect("x = {1}\nx -= 1", "unsupported operand type(s) for -=: 'set' and 'int'")
expect("x = {1}\nx &= 1", "unsupported operand type(s) for &=: 'set' and 'int'")

# The operands are named by their real class.  `@` reads them at the same
# layer every other operator does, so a user class is 'Foo' and not 'object'.
expect(
    "class Foo: pass\nx = Foo() @ Foo()",
    "unsupported operand type(s) for @: 'Foo' and 'Foo'",
)
expect(
    "class Foo: pass\nx = Foo()\nx @= Foo()",
    "unsupported operand type(s) for @=: 'Foo' and 'Foo'",
)
expect(
    "class Foo: pass\nx = Foo()\nx += 1",
    "unsupported operand type(s) for +=: 'Foo' and 'int'",
)

# An in-place special that exists but declines falls through to the binary
# dispatch, which still reports the augmented symbol.
expect(
    "class Foo:\n def __iand__(self, other): return NotImplemented\n"
    "x = Foo()\nx &= 1",
    "unsupported operand type(s) for &=: 'Foo' and 'int'",
)

# `operator.iand` and its siblings reach the same dispatch as `x &= y`, so
# they report the same symbol.
import operator

for name, symbol in [
    ("iadd", "+="), ("isub", "-="), ("iand", "&="), ("ior", "|="),
    ("ixor", "^="), ("ilshift", "<<="), ("irshift", ">>="),
    ("ifloordiv", "//="), ("imod", "%="), ("itruediv", "/="),
    ("ipow", "**="), ("imatmul", "@="),
]:
    try:
        getattr(operator, name)(1, "a")
    except TypeError as exc:
        assert str(exc) == (
            f"unsupported operand type(s) for {symbol}: 'int' and 'str'"
        ), f"operator.{name}: {exc}"
    else:
        raise AssertionError(f"operator.{name} did not raise TypeError")

# The operations that succeed are unaffected.
ns = {}
exec("x = 3\nx &= 6\ny = [1]\ny += [2]\nz = 2\nz **= 3\nw = 7\nw //= 2", ns)
assert ns["x"] == 2, ns["x"]
assert ns["y"] == [1, 2], ns["y"]
assert ns["z"] == 8, ns["z"]
assert ns["w"] == 3, ns["w"]

print("OK")
