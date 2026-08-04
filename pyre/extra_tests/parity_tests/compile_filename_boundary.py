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

SOURCE = "x = 1"


class Path:
    def __init__(self, name):
        self.name = name

    def __fspath__(self):
        return self.name


# A str filename is handed through unchanged.
assert compile(SOURCE, "plain.py", "exec").co_filename == "plain.py"

# bytes decode with the filesystem encoding.  The ASCII case matters on its own:
# it needs no surrogate to expose a filename that never reaches the code object.
assert compile(SOURCE, b"ascii.py", "exec").co_filename == "ascii.py"
assert compile(SOURCE, b"\xff.py", "exec").co_filename == "\udcff.py"

# A str already carrying a surrogate escape encodes back to that same byte
# instead of raising UnicodeEncodeError.
assert compile(SOURCE, "\udcff.py", "exec").co_filename == "\udcff.py"

# `__fspath__` is honoured, and may answer with either str or bytes.
assert compile(SOURCE, Path(b"\xff.py"), "exec").co_filename == "\udcff.py"
assert compile(SOURCE, Path("spelled.py"), "exec").co_filename == "spelled.py"

# An object that is neither a path nor a string is a TypeError, not a filename
# quietly replaced by "<string>".  The two runtimes word it differently, so only
# the class is under test.
for bad in (42, object(), None):
    try:
        compile(SOURCE, bad, "exec")
    except TypeError:
        pass
    else:
        raise AssertionError("compile() accepted %r as a filename" % (bad,))

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
    compile("(", b"\xff.py", "exec")
except SyntaxError as exc:
    assert exc.filename == "\udcff.py", ascii(exc.filename)
else:
    raise AssertionError("compile() accepted an unterminated '('")


# `repr()` reads `co_filename`, so it follows a replacement instead of showing
# the name the object was compiled under.
code = compile(SOURCE, "before.py", "exec")
assert 'file "before.py"' in repr(code), repr(code)

replaced = code.replace(co_filename="\udcff.py")
assert replaced.co_filename == "\udcff.py"
assert 'file "\udcff.py"' in repr(replaced), ascii(repr(replaced))

# `pycode.py:570-572` reports the zero sentinel as line -1.
assert 'line -1>' in repr(code.replace(co_firstlineno=0)), ascii(
    repr(code.replace(co_firstlineno=0))
)

# and a filename that never had a UTF-8 spelling reaches repr() the same way.
from_bytes = compile(SOURCE, b"\xff.py", "exec")
assert 'file "\udcff.py"' in repr(from_bytes), ascii(repr(from_bytes))


# Every reader of a code object agrees on the filename, so a frame built from
# one points at the real file rather than at "<string>".
namespace = {}
exec(compile("def f():\n    import sys\n    return sys._getframe()\n", b"\xff.py", "exec"), namespace)
frame = namespace["f"]()
assert frame.f_code.co_filename == "\udcff.py", ascii(frame.f_code.co_filename)
assert "�" not in repr(frame), ascii(repr(frame))

print("OK")
