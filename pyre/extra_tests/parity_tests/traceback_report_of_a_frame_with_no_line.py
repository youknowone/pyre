# CPython-suite gap: `test_traceback` renders plenty of tracebacks, but every
# frame in them sits on an instruction the compiler gave a position to, so the
# line slot always holds a number.  Nothing in the suite prints a frame whose
# line is missing, even though `co_positions()` reports those instructions and
# `tb_lineno` already answers `None` for them.
#
# parity-tests reason: the default report is written by the interpreter, not by
# the `traceback` module, so what it does with a `None` line is a separate
# implementation.  `FrameSummary.lineno` is whatever `tb_lineno` answered and
# is formatted straight into the `File "...", line ...` header, so the header
# reads `line None`; `_set_lines` returns early for a summary with no line, so
# no source line is echoed under it.  A report that substitutes a number there
# -- the resolver's `-1`, or the line that happened to precede the range --
# claims a position the code object does not have.
#
# PyPy 7.3.20 has no `TracebackType` constructor, so the child below fails
# there for the reason `traceback_lineno_sentinel.py` gives.

import subprocess
import sys
import tempfile
import os

PROBE = '''\
import sys
import types

SOURCE = """
def guarded(cm, box):
    box.append(sys._getframe())
    with cm:
        pass
"""
NAMESPACE = {"sys": sys}
exec(compile(SOURCE, "<no-line>", "exec"), NAMESPACE)
POSITIONS = list(NAMESPACE["guarded"].__code__.co_positions())
NO_LOCATION = [index for index, row in enumerate(POSITIONS) if row[0] is None]


class Manager:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


box = []
NAMESPACE["guarded"](Manager(), box)
error = ValueError("no line here")
error.__traceback__ = types.TracebackType(None, box[0], NO_LOCATION[0] * 2, -1)
raise error
'''

with tempfile.TemporaryDirectory() as folder:
    script = os.path.join(folder, "probe.py")
    with open(script, "w") as handle:
        handle.write(PROBE)
    child = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
    )

print("exit:", child.returncode)
for line in child.stderr.splitlines():
    # The child's own path is a temporary directory; only the rebuilt frame
    # and the exception line are being compared.
    if script in line or line.strip() == "raise error":
        continue
    print(repr(line))

print("OK")
