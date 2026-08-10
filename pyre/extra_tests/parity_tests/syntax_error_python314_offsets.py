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

print("syntax error Python 3.14 offsets ok")
