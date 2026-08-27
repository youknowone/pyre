# pyre-check: gate=1
# `PyCF_ALLOW_INCOMPLETE_INPUT` is read by the parser the mode selects, so the
# same unfinished source is classified differently in each of the three modes.
# `_IncompleteInputError` is what tells `codeop` to ask for another line; a
# mode whose parser has all the input there is reports the ordinary
# `SyntaxError` instead.

import codeop

ALLOW_INCOMPLETE_INPUT = 0x4000


def classify(source, mode):
    try:
        compile(source, "<test>", mode, ALLOW_INCOMPLETE_INPUT)
    except SyntaxError as error:
        return type(error).__name__
    return "OK"


# A single-quoted literal still open at the end of the source is reported as
# unterminated only by the block parser, which holds all the input there is.
# The interactive parser has another line to read, and `eval` wants one
# expression and stops at the end of the input it was given.
assert classify("x='a", "single") == "_IncompleteInputError"
assert classify("x='a", "exec") == "SyntaxError"
assert classify("'a", "exec") == "SyntaxError"
assert classify("'a", "eval") == "_IncompleteInputError"

# A triple-quoted one runs to the end of the source by construction.
assert classify("x='''a", "single") == "_IncompleteInputError"
assert classify("x='''a", "exec") == "_IncompleteInputError"
assert classify("'''a", "eval") == "_IncompleteInputError"

# `eval` parses one expression and stops at the first token that cannot
# continue it, so its tokenizer reaches the end of the source only when nothing
# earlier went wrong.  An unfinished expression is more input wanted...
for source in ("'a", "'''a", "[1,", "(1,", "1 +", "x{", "lambda", "not", "1 if 2", "1 1"):
    assert classify(source, "eval") == "_IncompleteInputError", source

# ...while a source that fails before its end is wrong where it stands, however
# much more input follows: each of these stops at the `=`, the keyword or the
# separator, not at the construct left open behind it.
for source in ("x='a", "x='''a", "x=[1,", "x=(1,", "x=1+", "x=1", "x=)",
               "if 1:", "if 1:\n", "def f():", "import os", "1;2"):
    assert classify(source, "eval") == "SyntaxError", source

# A source with no statement in it is unfinished for whichever parser wanted
# one, and `exec` wanted none.
assert classify("# hi\n", "eval") == "_IncompleteInputError"
assert classify("# hi\n", "exec") == "OK"
assert classify("", "eval") == "_IncompleteInputError"

# A program that is wrong where it stands is never unfinished.
for mode in ("exec", "eval", "single"):
    assert classify("x=)", mode) == "SyntaxError", mode

# `codeop` is the caller this classification exists for, and a caller that
# wants one expression asks it for `eval`.
assert codeop.compile_command("x='''a", "<test>", "single") is None
assert codeop.compile_command("x=1", "<test>", "single") is not None
assert codeop.compile_command("[1,", "<test>", "eval") is None
assert codeop.compile_command("[1]", "<test>", "eval") is not None

print("OK")
