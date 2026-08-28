# pyre-check: pypy-diverges: pypy3 states `embedded null byte` for the
# `path_converter` calls too, so the split this pins is not expressible there.
#
# CPython-suite gap: `test_os` and `test_builtin` assert the exception *type*
# for an embedded null and `test_compile` checks that null source is rejected,
# never with what wording, so a runtime that answers with a different reader's
# sentence passes every one.
#
# parity-tests reason: an embedded null is refused at three different readers
# and each states its own sentence.  `path_converter` prefixes the call it was
# reading (`stat: embedded null character in path`); the plain string converters
# behind `open`, `os.putenv` and `os.system` say `embedded null byte`; and the
# tokenizer refuses null source outright rather than treating it as one more
# unprintable character.  Which sentence a program sees says which reader
# stopped it, and a tokenizer that reports a position invites a fix to the
# source at that position when the whole string is the problem.
import os


def refusal(fn):
    try:
        fn()
    except BaseException as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    raise AssertionError("accepted")


NUL_BYTE = "ValueError: embedded null byte"

# `open` reads its path through the same converter for both spellings.
assert refusal(lambda: open("a\0b")) == NUL_BYTE, refusal(lambda: open("a\0b"))
assert refusal(lambda: open(b"a\0b")) == NUL_BYTE, refusal(lambda: open(b"a\0b"))
assert refusal(lambda: os.putenv("a\0b", "c")) == NUL_BYTE
assert refusal(lambda: os.system("a\0b")) == NUL_BYTE

# `path_converter` names the call instead, so these are a different sentence.
assert refusal(lambda: os.stat("a\0b")) == (
    "ValueError: stat: embedded null character in path"
), refusal(lambda: os.stat("a\0b"))
assert refusal(lambda: os.listdir("a\0b")) == (
    "ValueError: listdir: embedded null character in path"
)

# The tokenizer refuses the source, not a character in it.
assert refusal(lambda: compile("a\0b", "<s>", "exec")) == (
    "SyntaxError: source code string cannot contain null bytes"
), refusal(lambda: compile("a\0b", "<s>", "exec"))
assert refusal(lambda: compile("\0", "<s>", "exec")) == (
    "SyntaxError: source code string cannot contain null bytes"
)

# A two-path call carries both paths, and the second is what tells a program
# which end of the operation failed.
try:
    os.symlink("/tmp", "/tmp")
except OSError as exc:
    assert (exc.filename, exc.filename2) == ("/tmp", "/tmp"), (exc.filename, exc.filename2)
    assert str(exc).endswith("'/tmp' -> '/tmp'"), str(exc)
else:
    raise AssertionError("symlink over an existing path succeeded")

print("OK")
