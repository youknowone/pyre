# CPython-suite gap: test_syntax asserts the `cannot assign to ...` family by
# message text only, and never compiles a target whose operand is missing, so
# nothing in the suite reads what an incomplete target reports.
# parity-tests reason: CPython 3.14.6 and PyPy 7.3.20 answer the same message
# and the same offset for every case below, so all three runtimes are asserted
# against one literal.

"""An assignment target the parse never finished is not named.

`python.gram:invalid_assignment` names a target only once the target itself
parsed: `x == y = 1` is a complete comparison and is reported as one, over the
whole target.  `x ===` is not -- the comparison has no right operand -- so no
alternative matches and the failure is reported at the token the parse stopped
on, one column wide.

An external parser recovers instead of stopping, handing back a comparison
whose right operand is a synthesised node, and naming that target answers
`cannot assign to comparison` over `x ==` where both oracles answer
`invalid syntax` at the third `=`.  The recovery node is what makes the two
cases distinguishable, and the reported column is the token after it -- the
augmented form reports the operator's own two columns.

`end_offset` is asserted only where the token is inside the line: for a target
that runs off the end, the two oracles disagree on the end column while
agreeing on the start, so those cases assert the offset alone.
"""


def error(code):
    try:
        compile(code + "\n", "f.py", "exec")
    except SyntaxError as exc:
        return exc
    raise AssertionError(f"{code!r} did not raise SyntaxError")


def expect(code, message, offset, end_offset=None):
    exc = error(code)
    got = (exc.msg, exc.offset) if end_offset is None else (
        exc.msg,
        exc.offset,
        exc.end_offset,
    )
    want = (message, offset) if end_offset is None else (message, offset, end_offset)
    assert got == want, f"{code!r}\n  expected {want}\n  got      {got}"


# The target is a complete comparison, so it is named, over its own span.
expect("x == y = 1", "cannot assign to comparison", 1, 7)
expect("x < y = 1", "cannot assign to comparison", 1, 6)

# The comparison has no right operand: reported at the token after it.
expect("x ===", "invalid syntax", 5, 6)
expect("x == = 1", "invalid syntax", 6, 7)
expect("x == y == = 1", "invalid syntax", 11, 12)
expect("-x == = 1", "invalid syntax", 7, 8)
expect("not x == = 1", "invalid syntax", 10, 11)

# A binary operator with no right operand behaves the same way.
expect("x + = 1", "invalid syntax", 5, 6)
expect("x ** = 1", "invalid syntax", 6, 7)

# An augmented assignment reports the operator's own two columns.
expect("x == += 1", "invalid syntax", 6, 8)

# Inside a display or a subscript the stopping token is the closing bracket,
# not the assignment operator that follows it.
expect("[x ==] = 1", "invalid syntax", 6, 7)
expect("(x ==) = 1", "invalid syntax", 6, 7)
expect("x[1 ==] = 1", "invalid syntax", 7, 8)

# An unparenthesised tuple hands its elements over one at a time, so the
# incomplete element is what stops the parse.
expect("x, y == = 1", "invalid syntax", 9, 10)

# A target that runs off the end of the line is reported one past the text.
# The two oracles disagree on `end_offset` there, so only the offset is read.
expect("x ==", "invalid syntax", 5)
expect("f(x) ==", "invalid syntax", 8)
expect("a.b ==", "invalid syntax", 7)

# `eval` has no assignment rule at all, so every one of these is the plain
# failure at the operator whether the target finished or not.
for source in ["x ===", "x == = 1", "x + = 1"]:
    try:
        compile(source, "f.py", "eval")
    except SyntaxError as exc:
        assert exc.msg == "invalid syntax", f"eval {source!r}: {exc.msg}"
    else:
        raise AssertionError(f"eval {source!r} did not raise SyntaxError")

# A valid assignment is unaffected.
ns = {}
exec("x = 1\ny = x == 1\n[a, b] = (2, 3)\nc = {}\nc['k'] = 4", ns)
assert ns["y"] is True, ns["y"]
assert (ns["a"], ns["b"]) == (2, 3), (ns["a"], ns["b"])
assert ns["c"] == {"k": 4}, ns["c"]

print("OK")
