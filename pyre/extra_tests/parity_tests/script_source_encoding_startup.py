"""A script codec import runs inside the process ExecutionContext."""

import os
import subprocess
import sys
import tempfile


source = '# -*- coding: cp437 -*-\nvalue = "┬ó"\nprint(value)\n'
path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as script:
        path = script.name
        script.write(source.encode("cp437"))
    result = subprocess.run(
        [sys.executable, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().strip() == "┬ó", result.stdout
    assert b"panicked at" not in result.stderr, result.stderr
finally:
    if path is not None:
        os.unlink(path)

print("OK")
