# pyre-check: gate=1
# pyre-check: pypy-diverges: pypy3 reports U+0000; CPython 3.14's tokenizer names null bytes
"""A script file's tokenizer error retains the file location."""

import os
import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory() as directory:
    for name, source, lineno in (
        ("nul_lf.py", b"x = '\0' nothing to see here\n", 1),
        ("nul_cr.py", b"x = 1\rx = '\0' nothing to see here\r", 2),
        ("nul_crlf.py", b"x = 1\r\nx = '\0' nothing to see here\r\n", 2),
    ):
        path = os.path.join(directory, name)
        with open(path, "wb") as stream:
            stream.write(source)

        result = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode != 0, result.stdout
        stderr = result.stderr.decode("utf-8", "replace")
        assert 'File "%s", line %d' % (path, lineno) in stderr, stderr
        assert stderr.splitlines()[-2:] == [
            "    x = '",
            "SyntaxError: source code cannot contain null bytes",
        ], stderr
        assert "\0" not in stderr, stderr

print("OK")
