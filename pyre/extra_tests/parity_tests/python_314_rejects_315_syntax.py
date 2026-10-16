# CPython-suite gap: the suite does not pin 3.14 SyntaxError wording and
# offsets for PEP 810 `lazy import`, PEP 798 unpacking in a comprehension,
# or a unary-plus match value.
# parity-tests reason: CPython 3.14.6 rejects each snippet below; a compiler
# pin that parses with no target version would accept 3.15 grammar.

"""3.14 compile() rejects syntax added in 3.15."""


def error(source):
    try:
        compile(source, "f.py", "exec")
    except SyntaxError as exc:
        return exc
    raise AssertionError(f"{source!r} did not raise SyntaxError")


def expect(source, message, lineno, offset, end_lineno, end_offset):
    exc = error(source)
    got = (exc.msg, exc.lineno, exc.offset, exc.end_lineno, exc.end_offset)
    want = (message, lineno, offset, end_lineno, end_offset)
    assert got == want, f"{source!r}\n  expected {want}\n  got      {got}"


expect("lazy import os", "invalid syntax", 1, 6, 1, 12)
expect("lazy from os import path", "invalid syntax", 1, 6, 1, 10)
expect(
    "[*x for x in [[1],[2]]]",
    "iterable unpacking cannot be used in comprehension",
    1, 2, 1, 4,
)
expect(
    "{*x for x in [[1],[2]]}",
    "iterable unpacking cannot be used in comprehension",
    1, 2, 1, 4,
)
expect(
    "(*x for x in [[1],[2]])",
    "iterable unpacking cannot be used in comprehension",
    1, 2, 1, 4,
)
expect(
    "f(*x for x in [[1],[2]])",
    "iterable unpacking cannot be used in comprehension",
    1, 3, 1, 5,
)
expect(
    "{**d for d in [{1:2}]}",
    "dict unpacking cannot be used in dict comprehension",
    1, 2, 1, 4,
)
expect("match 1:\n case +1: pass", "invalid syntax", 2, 7, 2, 8)

# A 3.14 form that must still compile.
ns = {}
exec("import sys\nx = [a for a in (1, 2)]\nmatch 1:\n case 1: y = 1\n", ns)
assert ns["x"] == [1, 2], ns["x"]
assert ns["y"] == 1, ns["y"]

print("OK")
