"""Python 3.14 codeop distinguishes incomplete input from invalid syntax.

CPython's ``Lib/codeop.py:64-81`` catches the private builtin
``_IncompleteInputError`` produced under ``PyCF_ALLOW_INCOMPLETE_INPUT`` and
returns ``None`` only for source which can become complete with more input.
"""

import builtins
import codeop


assert issubclass(builtins._IncompleteInputError, SyntaxError)

for source in ("if True:", "(", "'''unterminated"):
    assert codeop.compile_command(source) is None

try:
    codeop.compile_command("value =")
except SyntaxError as exc:
    assert type(exc) is SyntaxError
else:
    raise AssertionError("invalid syntax was classified as incomplete")

compiled = codeop.compile_command("value = 42")
assert compiled is not None

print("OK")
