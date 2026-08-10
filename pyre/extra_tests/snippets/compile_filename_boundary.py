# pyre-check: platforms=linux,darwin
# CPython-suite gap: compile() does not cover this filesystem-codec boundary.
# It is generic compiler behavior, so it belongs in snippets.
# `b"\xff.py"` only survives the filesystem-encoding converter where that
# encoding is UTF-8 with `surrogateescape`; on Windows the same call raises
# UnicodeDecodeError in the reference too.
"""`compile()` takes its filename through the filesystem-encoding converter.

`compiling.py:13` declares `@unwrap_spec(filename='fsencode', mode='text')`, so
the filename accepts `bytes` and `os.PathLike` as well as `str`, keeps a byte
with no UTF-8 spelling, and rejects an embedded NUL — while a non-`str` mode is
a TypeError rather than a silent fall back to "exec".  Every code object reader
then has to report that filename, including `repr()`, which reads the same
field `co_filename` does rather than a separate compiler-side spelling.

A `bytearray` filename is deliberately absent: it is a readable buffer that
`baseobjspace.py:1975` accepts with a DeprecationWarning but CPython rejects,
and this suite has to pass under CPython too.
"""

import sys

SOURCE = "x = 1"

# The filename the filesystem encoding carries but plain UTF-8 text cannot, and
# the bytes that spell it.  Windows encodes with `surrogatepass` (PEP 529), so
# the lone surrogate is spelled by its own three UTF-8 bytes and a byte that
# begins no sequence has no spelling at all; every other platform uses
# `surrogateescape`, where that same byte is what the surrogate stands for.
if sys.platform == "win32":
    NAME_BYTES, NAME_TEXT = b"\xed\xb3\xbf.py", "\udcff.py"
    UNSPELLABLE = b"\xff.py"
else:
    NAME_BYTES, NAME_TEXT = b"\xff.py", "\udcff.py"
    UNSPELLABLE = None


class Path:
    def __init__(self, name):
        self.name = name

    def __fspath__(self):
        return self.name


# Bytes decode with the filesystem encoding.
assert compile(SOURCE, NAME_BYTES, "exec").co_filename == NAME_TEXT

# A str the encoding can only spell as a surrogate encodes back to the same
# bytes instead of raising UnicodeEncodeError.
assert compile(SOURCE, NAME_TEXT, "exec").co_filename == NAME_TEXT

# Bytes the encoding cannot spell are reported, not renamed.
if UNSPELLABLE is not None:
    try:
        compile(SOURCE, UNSPELLABLE, "exec")
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError("compile() invented a spelling for %r" % (UNSPELLABLE,))

# The filesystem boundary also applies to a bytes-valued `__fspath__`.
assert compile(SOURCE, Path(NAME_BYTES), "exec").co_filename == NAME_TEXT

# `bytesbuf0_w` rejects an embedded NUL.
for nul in ("a\x00b.py", b"a\x00b.py"):
    try:
        compile(SOURCE, nul, "exec")
    except ValueError:
        pass
    else:
        raise AssertionError("compile() accepted a NUL in %r" % (nul,))

# mode='text' — a non-str mode is rejected rather than treated as "exec".
try:
    compile(SOURCE, "a.py", 42)
except TypeError:
    pass
else:
    raise AssertionError("compile() accepted a non-str mode")


# The filename a SyntaxError reports is the same one the successful compile
# would have recorded.
try:
    compile("(", NAME_BYTES, "exec")
except SyntaxError as exc:
    assert exc.filename == NAME_TEXT, ascii(exc.filename)
else:
    raise AssertionError("compile() accepted an unterminated '('")


# `repr()` reads `co_filename`, so it follows a surrogate-bearing replacement.
code = compile(SOURCE, "before.py", "exec")
replaced = code.replace(co_filename=NAME_TEXT)
assert replaced.co_filename == NAME_TEXT
assert 'file "%s"' % NAME_TEXT in repr(replaced), ascii(repr(replaced))

# and a filename that has no plain-text spelling reaches repr() the same way.
from_bytes = compile(SOURCE, NAME_BYTES, "exec")
assert 'file "%s"' % NAME_TEXT in repr(from_bytes), ascii(repr(from_bytes))


# Every reader of a code object agrees on the filename, so a frame built from
# one points at the real file rather than at "<string>".
namespace = {}
exec(
    compile("def f():\n    import sys\n    return sys._getframe()\n", NAME_BYTES, "exec"),
    namespace,
)
frame = namespace["f"]()
assert frame.f_code.co_filename == NAME_TEXT, ascii(frame.f_code.co_filename)
assert "�" not in repr(frame), ascii(repr(frame))

print("OK")
