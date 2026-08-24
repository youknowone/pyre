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

# Not covered here: CPython rejects a type comment the grammar has no place
# for — `x += 1  # type: int` and `f()  # type: int` are both
# `SyntaxError` there — and pyre accepts them and attaches nothing.  That is a
# narrower gap than dropping every comment, and it is left open.

print("OK")
