# CPython-suite gap: source-codec startup is not run through pyre's process EC.
# parity-tests reason: guard codec imports during script execution startup.

"""A script codec import runs inside the process ExecutionContext."""

import os
import subprocess
import sys
import tempfile


source = (
    '# -*- coding: cp437 -*-\n'
    'value = "┬ó"\n'
    'assert tuple(map(ord, value)) == (0x252c, 0xf3)\n'
    'print("OK")\n'
)
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
    assert result.stdout.strip() == b"OK", result.stdout
    assert b"panicked at" not in result.stderr, result.stderr
finally:
    if path is not None:
        os.unlink(path)

print("OK")
