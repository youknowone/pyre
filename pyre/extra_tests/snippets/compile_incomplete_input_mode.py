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

# A single-quoted literal still open at the end of the source is more input
# only for the interactive parser; the block parsers report it as unterminated.
assert classify("x='a", "single") == "_IncompleteInputError"
assert classify("x='a", "exec") == "SyntaxError"
assert classify("x='a", "eval") == "SyntaxError"

# A triple-quoted one runs to the end of the source by construction.
assert classify("x='''a", "single") == "_IncompleteInputError"
assert classify("x='''a", "exec") == "_IncompleteInputError"

# `eval` parses one expression and reaches its own end-of-input error ahead of
# the flag, so nothing that stops mid-expression is unfinished input there.
assert classify("x=[1,", "eval") == "SyntaxError"
assert classify("if 1:\n", "eval") == "SyntaxError"

# A source with no statement in it is unfinished for whichever parser wanted
# one, and `exec` wanted none.
assert classify("# hi\n", "eval") == "_IncompleteInputError"
assert classify("# hi\n", "exec") == "OK"

# A program that is wrong where it stands is never unfinished.
for mode in ("exec", "eval", "single"):
    assert classify("x=)", mode) == "SyntaxError", mode

# `codeop` is the caller this classification exists for.
assert codeop.compile_command("x='''a", "<test>", "single") is None
assert codeop.compile_command("x=1", "<test>", "single") is not None

print("OK")
