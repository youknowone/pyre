# pyre-check: gate=1
# CPython-suite gap: test_type_comments is not in the gated set, and nothing
# else in the suite passes `type_comments=True`, so a parse that accepts the
# flag and collects nothing looks exactly like one that works.
# parity-tests reason: pyre took the flag, parsed, and returned a tree whose
# `type_comment` was None everywhere and whose `type_ignores` was empty. The
# comments never reached the tree because the parser's token list was dropped.

# pyre-check: pypy-diverges: pypy3 collects `type_comment` but rejects a
# `# type: ignore` line outright with `SyntaxError: invalid syntax`.

import ast


def parse(src):
    return ast.parse(src, type_comments=True)


def comments(src):
    """Every node that carries a type comment, as (class, lineno, text)."""
    found = [
        (type(n).__name__, getattr(n, "lineno", None), n.type_comment)
        for n in ast.walk(parse(src))
        if getattr(n, "type_comment", None) is not None
    ]
    return sorted(found, key=lambda row: (row[1] or 0, row[0], row[2]))


def ignores(src):
    return [(i.lineno, i.tag) for i in parse(src).type_ignores]


# `lexer.c` matches the comment against `"# type: "`, where each space in the
# pattern stands for any run of spaces and tabs.
assert comments("x = 1  # type: int\n") == [("Assign", 1, "int")]
assert comments("x = 1  #type:int\n") == [("Assign", 1, "int")]
assert comments("x = 1  #\ttype:\tint\n") == [("Assign", 1, "int")]
# The trailing space matches an empty run, so a bare `# type:` still counts.
assert comments("x = 1  # type:\n") == [("Assign", 1, "")]
# What follows the prefix is taken verbatim, trailing spaces and all, and a
# second `# type:` inside it is just more text.
assert comments("x = 1  # type: int   \n") == [("Assign", 1, "int   ")]
assert comments("x = 1  # type: int  # type: str\n") == [
    ("Assign", 1, "int  # type: str")
]
# An ordinary comment stays ordinary, and one inside a string is not a comment.
assert comments("x = 1  # ordinary\n") == []
assert comments("x = '# type: int'\n") == []

# The grammar takes a type comment after an assignment's value — through a
# parenthesised value that ends on a later line, and at any nesting.
assert comments("x = y = 1  # type: int\n") == [("Assign", 1, "int")]
assert comments("x = (\n    1\n)  # type: int\n") == [("Assign", 1, "int")]
assert comments("if 1:\n    x = 1  # type: int\n") == [("Assign", 2, "int")]
assert comments("class C:\n    x = 1  # type: int\n") == [("Assign", 2, "int")]
assert comments("while 1:\n    x = 1  # type: int\n") == [("Assign", 2, "int")]
assert comments("try:\n    x = 1  # type: int\nexcept E:\n    y = 2  # type: str\n") == [
    ("Assign", 2, "int"),
    ("Assign", 4, "str"),
]

# ...after the colon of a `for`, a `with` or a `def`, including the async
# spellings, which are separate classes in the tree.
assert comments("for i in []:  # type: int\n    pass\n") == [("For", 1, "int")]
assert comments("with open('f') as g:  # type: IO\n    pass\n") == [("With", 1, "IO")]
assert comments("with a as b, c as d:  # type: IO\n    pass\n") == [("With", 1, "IO")]
assert comments("def f(a, b):  # type: (int, str) -> None\n    pass\n") == [
    ("FunctionDef", 1, "(int, str) -> None")
]
assert comments("def f(a) -> int:  # type: (int) -> int\n    pass\n") == [
    ("FunctionDef", 1, "(int) -> int")
]
# A function's comment may also stand on its own line before the body.
assert comments("def f(a):\n    # type: (int) -> int\n    pass\n") == [
    ("FunctionDef", 1, "(int) -> int")
]
assert comments("async def g(a):  # type: (int) -> None\n    pass\n") == [
    ("AsyncFunctionDef", 1, "(int) -> None")
]
assert comments("async def g():\n    async for i in []:  # type: int\n        pass\n") == [
    ("AsyncFor", 2, "int")
]
assert comments("async def g():\n    async with a as b:  # type: IO\n        pass\n") == [
    ("AsyncWith", 2, "IO")
]

# ...and after a parameter's comma, which is a different span from the
# function's own: one is inside the parentheses and one is past them.
assert comments("def f(\n    a,  # type: int\n    b,  # type: str\n):\n    pass\n") == [
    ("arg", 2, "int"),
    ("arg", 3, "str"),
]
assert comments("def f(\n    a=1,  # type: int\n):\n    pass\n") == [("arg", 2, "int")]
assert comments("def f(\n    *args,  # type: int\n    **kw,  # type: str\n):\n    pass\n") == [
    ("arg", 2, "int"),
    ("arg", 3, "str"),
]
assert comments(
    "def f(\n    a,  # type: int\n):\n    # type: (int) -> None\n    pass\n"
) == [("FunctionDef", 1, "(int) -> None"), ("arg", 2, "int")]

# `# type: ignore` is the word `ignore` followed by the end of the comment or
# by an ASCII byte that is not alphanumeric.  `_` and `-` qualify; a letter
# does not, and neither does a non-ASCII byte.
assert ignores("x = 1  # type: ignore[foo]\n") == [(1, "[foo]")]
assert ignores("x = 1  # type: ignore_foo\n") == [(1, "_foo")]
assert ignores("x = 1  # type: ignore-foo\n") == [(1, "-foo")]
assert ignores("x = 1  # type:  ignore  extra\n") == [(1, "  extra")]
assert ignores("x = 1  # type: ignorexyz\n") == []
assert comments("x = 1  # type: ignorexyz\n") == [("Assign", 1, "ignorexyz")]
assert ignores("x = 1  # type: ignore\u00e4\n") == []
# An ignore that stands alone on its line takes the line break with it.
assert ignores("# type: ignore\nx = 1\n") == [(1, "\n")]
assert ignores("# type: ignore  extra\nx = 1\n") == [(1, "  extra\n")]
assert ignores("# type: ignore\n# type: ignore[a]\nx = 1\n") == [(1, "\n"), (2, "[a]\n")]
assert ignores("def f():\n    # type: ignore\n    pass\n") == [(2, "\n")]
# One sharing its line with code does not.
assert ignores("def f(a):  # type: ignore\n    pass\n") == [(1, "")]

# The two kinds are collected from one pass over the same comments.
mixed = "# type: ignore\nx = 1  # type: int\nfor i in []:  # type: str\n    pass\n"
assert ignores(mixed) == [(1, "\n")]
assert comments(mixed) == [("Assign", 2, "int"), ("For", 3, "str")]

# Without the flag nothing is collected, and the flag does not change the tree
# it is collected from.
plain = ast.parse(mixed)
assert plain.type_ignores == []
assert [n.type_comment for n in ast.walk(plain) if hasattr(n, "type_comment")] == [
    None,
    None,
]
assert ast.dump(plain) != ast.dump(parse(mixed))

# An assignment reads its comment straight after the value, so the comment has
# to be the next token rather than merely the last thing on the line.  With
# `;` between them it belongs to the assignment it follows, and to no one when
# nothing follows it.


def owner(src):
    """(class, col_offset, text) of every node carrying a type comment."""
    return sorted(
        (type(n).__name__, n.col_offset, n.type_comment)
        for n in ast.walk(parse(src))
        if getattr(n, "type_comment", None) is not None
    )


assert owner("y = 1; z = 2  # type: int") == [("Assign", 7, "int")]
assert owner("y = 1; z = 2; w = 3  # type: int") == [("Assign", 14, "int")]
# A block written on one line puts the body where the comment would go, so the
# statement inside takes it and the header does not.
assert owner("for i in []: x = 1  # type: int") == [("Assign", 13, "int")]
assert owner("with a as b: x = 1  # type: int") == [("Assign", 13, "int")]
assert owner("def f(): x = 1  # type: int") == [("Assign", 9, "int")]
assert owner("if x: y = 1  # type: int") == [("Assign", 6, "int")]
assert owner("while x: y = 1  # type: int") == [("Assign", 9, "int")]
# A line break is not the whitespace the rule tolerates.
assert owner("x = (1 +\n     2)  # type: int") == [("Assign", 0, "int")]
assert owner("x = 1 + \\\n    2  # type: int") == [("Assign", 0, "int")]

# A `TYPE_COMMENT` is a token, so a rule that does not accept one leaves the
# parser looking at a token it cannot shift.  Collecting the comments is only
# half the flag: the other half is refusing the positions the grammar has no
# rule for, and the refusal is the parser's ordinary `invalid syntax`.


def refused(src, mode="exec"):
    """(lineno, offset, end_offset, text) of the SyntaxError, or None."""
    try:
        ast.parse(src, mode=mode, type_comments=True)
    except SyntaxError as e:
        return (e.lineno, e.offset, e.end_offset, e.text)
    return None


# The five accepted positions stay accepted.
for accepted in (
    "x = 1  # type: int\n",
    "x = y = 1  # type: int\n",
    "for i in []:  # type: int\n    pass\n",
    "with a as b:  # type: int\n    pass\n",
    "def f():  # type: () -> int\n    pass\n",
    "def f(a,  # type: int\n      ):\n    pass\n",
    "x += 1  # type: ignore\n",
):
    assert refused(accepted) is None, accepted

# Everything else is refused, including an annotated assignment — the rule
# that takes a comment is the plain `=` one.
assert refused("x += 1  # type: int") == (1, 17, 20, "x += 1  # type: int\n")
assert refused("f()  # type: int") == (1, 14, 17, "f()  # type: int\n")
assert refused("x: int = 1  # type: int") == (1, 21, 24, "x: int = 1  # type: int\n")
assert refused("pass  # type: int") == (1, 15, 18, "pass  # type: int\n")
for one_liner in (
    "if x:  # type: int\n    pass\n",
    "while x:  # type: int\n    pass\n",
    "class C:  # type: int\n    pass\n",
    "import os  # type: int\n",
    "del x  # type: int\n",
    "assert x  # type: int\n",
    "raise E  # type: int\n",
    "global x  # type: int\n",
    "def f():\n    return 1  # type: int\n",
):
    assert refused(one_liner) is not None, one_liner

# The same adjacency decides a refusal: with nothing the rule accepts standing
# directly before it, the comment is a token no rule takes.
assert refused("y = 1; z += 1  # type: int") == (1, 24, 27, "y = 1; z += 1  # type: int\n")
assert refused("y = 1;  # type: int") == (1, 17, 20, "y = 1;  # type: int\n")
# A block written on one line leaves the header's slot occupied by the body.
assert refused("for i in []: pass  # type: int") is not None
assert refused("def f(): pass  # type: int") is not None

# A comment standing on a line of its own is a token too, and the line the
# tokenizer reports carries the break a source without one is missing.
assert refused("# type: int\nx = 1") == (1, 9, 12, "# type: int\n")
assert refused("x = 1\n# type: int") == (2, 9, 12, "# type: int\n")

# The span is the token, which begins past the prefix however the prefix is
# spelled, and ends where the comment does.
assert refused("x += 1\t#\ttype:\tint") == (1, 16, 19, "x += 1\t#\ttype:\tint\n")
assert refused("x += 1  #type:int") == (1, 15, 18, "x += 1  #type:int\n")
assert refused("x += 1  # type:") == (1, 16, 16, "x += 1  # type:\n")
# `ignore` has to stand on its own to be an ignore; `ignoreX` is a type.
assert refused("x += 1  # type: ignoreX") == (1, 17, 24, "x += 1  # type: ignoreX\n")

# The offset counts characters, not the bytes they encode.
assert refused("y = '\uc548\ub155'; z += 1  # type: int")[1] == 27

# The parser stops at the first token it cannot shift, so an earlier accepted
# comment does not save a later refused one, and an earlier refused one is
# what gets reported.
assert refused("x = 1  # type: int\ny += 2  # type: str")[0] == 2
assert refused("x += 1  # type: int\ny = 2  # type: str")[0] == 1

# An expression has none of the five positions, so `mode="eval"` refuses any
# comment at all.  File input has a line break appended by the tokenizer, so
# the line it reports carries one; `eval` and `single` report it as it stands.
assert refused("x  # type: int", mode="eval") == (1, 12, 15, "x  # type: int")
assert refused("x  # type: int\n", mode="eval") == (1, 12, 15, "x  # type: int\n")
assert refused("x += 1  # type: int", mode="single") == (
    1,
    17,
    20,
    "x += 1  # type: int",
)
assert refused("x", mode="eval") is None

# `test_type_comments.py::test_inappropriate_type_comments` is the list the
# suite grades a runtime against; every one of its nine is refused here, at the
# same place.
assert refused("pass  # type: int\n") == (1, 15, 18, "pass  # type: int\n")
assert refused("foo()  # type: int\n") == (1, 16, 19, "foo()  # type: int\n")
assert refused("x += 1  # type: int\n") == (1, 17, 20, "x += 1  # type: int\n")
assert refused("while True:  # type: int\n  continue\n")[:3] == (1, 22, 25)
assert refused("while True:\n  continue  # type: int\n")[:3] == (2, 21, 24)
assert refused("try:  # type: int\n  pass\nfinally:\n  pass\n")[:3] == (1, 15, 18)
assert refused("try:\n  pass\nfinally:  # type: int\n  pass\n")[:3] == (3, 19, 22)
# `ignore` only stands on its own when what follows is not alphanumeric, so
# both of these are types rather than ignores -- including the non-ASCII one,
# which the byte test has to treat as a letter.
assert refused("pass  # type: ignorewhatever\n")[:3] == (1, 15, 29)
assert refused("pass  # type: ignore\u00e9\n")[:3] == (1, 15, 22)

# A comment on the line after the header is the function's own -- the rule
# reads `NEWLINE TYPE_COMMENT` before the block.
assert owner("def f():\n    # type: () -> str\n    pass\n") == [
    ("FunctionDef", 0, "() -> str"),
]
# Taking one on the header and finding another before the body is the one
# failure the grammar names for itself, and it reports the `INDENT`: the
# body's own indent as a column, and no end.
for indent in ("  ", "    ", "        "):
    src = "def f():  # type: () -> a\n%s# type: () -> b\n%spass\n" % (indent, indent)
    assert refused(src) == (3, len(indent), -1, indent + "pass\n"), indent
# A blank line between them does not change which rule matches.
assert refused("def f():  # type: () -> a\n\n    # type: () -> b\n    pass\n") == (
    4,
    4,
    -1,
    "    pass\n",
)
# The body's comment is its statement's, so it is not a second one on the def.
assert owner("def f():  # type: () -> a\n    x = 1  # type: int\n    pass\n") == [
    ("Assign", 4, "int"),
    ("FunctionDef", 0, "() -> a"),
]

# Without the flag the comment is just a comment.
ast.parse("x += 1  # type: int")

print("OK")
