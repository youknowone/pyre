"""Every reader of `co_filename` reports the byte-exact filesystem spelling.

`pycode.py:135 self.co_filename = filename` keeps the raw path, so the frame
repr (`pyframe.py:849-853 descr_repr`), the warning location, a coroutine's
origin and the rendered traceback all name what the `co_filename` getter hands
out — a path byte with no UTF-8 spelling must not fold to U+FFFD on the way.
"""

import _imp
import io
import subprocess
import sys
import traceback
import warnings

FILENAME = "x\udcff.py"


def stamped(source):
    """A code object carrying a filename no UTF-8 encoder can spell."""
    code = compile(source, "orig.py", "exec")
    _imp._fix_co_filename(code, FILENAME)
    return code


# The stdlib formatter reads `co_filename` directly, so this is the anchor the
# rest are measured against.
try:
    exec(stamped("raise ValueError('boom')"), {})
except ValueError:
    buf = io.StringIO()
    traceback.print_exc(file=buf)
    text = buf.getvalue()
assert FILENAME in text, ascii(text)
assert "�" not in text, ascii(text)

# `descr_repr` interpolates the filename rather than its repr, so the surrogate
# reaches the result unescaped; CPython formats that field with `%R` and spells
# it `x\udcff.py`.  Both keep the byte, so only the folding is asserted here.
namespace = {"sys": sys}
exec(stamped("def capture():\n    return sys._getframe()\nframe = capture()"), namespace)
frame_repr = repr(namespace["frame"])
assert "�" not in frame_repr, ascii(frame_repr)

# The warning's location comes from the frame's own code object.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    exec(stamped("import warnings\nwarnings.warn('w')"), {})
assert caught[0].filename == FILENAME, ascii(caught[0].filename)

# `cr_origin` summarises the same frames.
sys.set_coroutine_origin_tracking_depth(2)
namespace = {}
try:
    exec(stamped("async def c():\n    pass\ncoro = c()"), namespace)
    origin = namespace["coro"].cr_origin
    namespace["coro"].close()
finally:
    sys.set_coroutine_origin_tracking_depth(0)
assert origin[0][0] == FILENAME, ascii(origin[0][0])

# The unhandled-exception renderer writes to stderr with no text layer in
# between, so the name arrives as the filesystem bytes and decodes back with
# the same error handler that produced it.
if sys.platform != "win32":
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import _imp\n"
            "code = compile(\"raise ValueError('boom')\", 'orig.py', 'exec')\n"
            "_imp._fix_co_filename(code, 'x\\udcff.py')\n"
            "exec(code)\n",
        ],
        capture_output=True,
    )
    stderr = child.stderr.decode(errors="surrogateescape")
    assert "�" not in stderr, ascii(stderr)

print("OK")
