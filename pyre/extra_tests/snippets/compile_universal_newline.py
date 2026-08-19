# pyre-check: gate=1
"""A source's line terminators are all `\n` by the time the tokenizer sees it.

`pytokenizer.py:654-662` universal_newline rewrites a line ending in `\r\n` or
in a lone `\r` to one ending in `\n`, and `generate_tokens` calls it on every
line it takes from `splitlines(True)` (`pyparse.py:202`).  So the rewrite is not
string syntax and not a property of one entry point: it reaches a file, a
`compile()` argument, `ast.parse` and `-c` alike, and the text of a
triple-quoted literal along with the code around it.

Neither rewrite moves a line boundary, so the line a statement reports is the
one it reported before.
"""

import ast
import os
import subprocess
import sys
import tempfile

CRLF = "x = \"\"\"a\r\nb\"\"\""
LONE_CR = "x = \"\"\"a\rb\"\"\""


def value_of(source, name="x"):
    namespace = {}
    exec(compile(source, "<test>", "exec"), namespace)
    return namespace[name]


# The literal spans the terminator, so its own text carries the rewrite.
assert value_of(CRLF) == "a\nb", repr(value_of(CRLF))
assert value_of(LONE_CR) == "a\nb", repr(value_of(LONE_CR))

# A terminator is not an escape, so being raw or bytes changes nothing.
assert value_of("x = r\"\"\"a\rb\"\"\"") == "a\nb", repr(value_of("x = r\"\"\"a\rb\"\"\""))
assert value_of("x = b\"\"\"a\r\nb\"\"\"") == b"a\nb", repr(value_of("x = b\"\"\"a\r\nb\"\"\""))

# `ast.parse` reads the same rewritten source, so the constant it carries and
# the segment `end_col_offset` describes agree with it.
tree = ast.parse(CRLF)
assert tree.body[0].value.value == "a\nb", repr(tree.body[0].value.value)
assert ast.parse(LONE_CR).body[0].value.value == "a\nb"

# Statements still land on the lines they were written on.
namespace = {}
exec(compile("x = 1\r\ny = 2\r\nz = 3\r", "<test>", "exec"), namespace)
assert (namespace["x"], namespace["y"], namespace["z"]) == (1, 2, 3), namespace
try:
    exec(compile("x = 1\rraise ValueError('boom')", "<test>", "exec"), {})
except ValueError as exc:
    assert exc.__traceback__.tb_next.tb_lineno == 2, exc.__traceback__.tb_next.tb_lineno
else:
    raise AssertionError("the raise did not run")

# A failed compile reports the offending line, and that line came from the
# rewritten source too.  How the reference spells the terminator is its own
# business -- CPython drops it and PyPy keeps `\n` -- so what is pinned here is
# only that no carriage return survives into it.
try:
    compile("x = 1\r\ny = (\r\nz = 3\r\n", "<test>", "exec")
except SyntaxError as exc:
    assert exc.lineno == 2, exc.lineno
    assert exc.text is not None and "\r" not in exc.text, repr(exc.text)
    assert exc.text.rstrip("\r\n") == "y = (", repr(exc.text)
else:
    raise AssertionError("the unclosed paren did not raise")

# A lone `\r` reaches the same slicing.  Before the rewrite ran on the string
# the report is sliced from, this source held no `\n` at all, so the offending
# line came back as `None` rather than as the second line.
try:
    compile("a = 1\rb = (\r", "<test>", "exec")
except SyntaxError as exc:
    assert exc.lineno == 2, exc.lineno
    assert exc.text is not None and "\r" not in exc.text, repr(exc.text)
    assert exc.text.rstrip("\r\n") == "b = (", repr(exc.text)
else:
    raise AssertionError("the unclosed paren did not raise")

# The same source reaching the compiler as a `-c` argument, and as a file.
PROGRAM = CRLF + "\nprint(repr(x))\n"

completed = subprocess.run(
    [sys.executable, "-c", PROGRAM],
    capture_output=True,
    text=True,
)
assert completed.returncode == 0, completed.stderr
assert completed.stdout.strip() == "'a\\nb'", completed.stdout

# Written as bytes so the carriage returns reach the file itself rather than
# whatever the platform's text mode would spell them as.
handle, path = tempfile.mkstemp(suffix=".py")
os.write(handle, PROGRAM.encode())
os.close(handle)
try:
    completed = subprocess.run([sys.executable, path], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "'a\\nb'", completed.stdout
finally:
    os.unlink(path)

print("OK")
