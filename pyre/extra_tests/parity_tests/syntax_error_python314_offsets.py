"""Python 3.14 SyntaxError positions use character, not UTF-8 byte, columns."""


def syntax_error(source):
    try:
        compile(source, "<fragment>", "exec")
    except SyntaxError as error:
        return error
    raise AssertionError("source unexpectedly compiled")


error = syntax_error('Python = "Ṕýţĥòñ" +')
assert (error.lineno, error.offset) == (1, 20), (
    error.lineno,
    error.offset,
)

error = syntax_error('α = 0xI')
assert (error.lineno, error.offset) == (1, 6), (
    error.lineno,
    error.offset,
)

error = syntax_error(b'\n\n\nPython = "\xcf\xb3\xf2\xee\xed" +')
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (4, 12, 4, 12), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)
assert error.text == 'Python = "ϳ���" +', error.text

error = syntax_error(b"\xef\xbb\xbf#coding: utf8\nprint('\xe6\x88\x91')\n")
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 0, 1, 13), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)
assert error.text == "#coding: utf8", error.text

error = syntax_error('return "ä"')
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 1, 1, 12), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

error = syntax_error('x = rb"é"')
assert error.msg == "bytes can only contain ASCII literal characters", error.msg
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 5, 1, 10), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

for source, expected in (
    ("f'{6 0}'", (1, 4, 1, 7)),
    ('f"""\n\n\n            {\n            6\n            0="""', (5, 13, 6, 14)),
):
    error = syntax_error(source)
    assert error.msg == "invalid syntax. Perhaps you forgot a comma?", error.msg
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == expected, (
        error.lineno,
        error.offset,
        error.end_lineno,
        error.end_offset,
    )

error = syntax_error('if True:\nprint "No indent"')
assert isinstance(error, IndentationError), type(error)
assert error.msg == "expected an indented block after 'if' statement on line 1", error.msg
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (2, 1, 2, 6), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

error = syntax_error('if True:\n        print()\n\texec "mixed tabs and spaces"')
assert isinstance(error, TabError), type(error)
assert error.msg == "inconsistent use of tabs and spaces in indentation", error.msg

error = syntax_error("def f():\n  global x\n  nonlocal x")
assert error.msg == "name 'x' is nonlocal and global", error.msg
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (2, 3, 2, 11), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

error = syntax_error('"┬ó┬ó┬ó┬ó┬ó┬ó" + f(4, x for x in range(1))')
assert error.msg == "invalid syntax", error.msg
assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == (1, 25, 1, 28), (
    error.lineno,
    error.offset,
    error.end_lineno,
    error.end_offset,
)

print("syntax error Python 3.14 offsets ok")
