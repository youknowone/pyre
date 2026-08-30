# pyre-check: gate=1
# pyre-check: pypy-diverges: pypy3 reports U+0000; CPython 3.14's tokenizer names null bytes
"""A script file's tokenizer error retains the file location."""

import os
import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "nul.py")
    with open(path, "wb") as stream:
        stream.write(b"x = '\0' nothing to see here\n")

    result = subprocess.run(
        [sys.executable, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode != 0, result.stdout
    stderr = result.stderr.decode("utf-8", "replace")
    assert 'File "%s", line 1' % path in stderr, stderr
    assert "SyntaxError: source code cannot contain null bytes" in stderr, stderr

print("OK")
