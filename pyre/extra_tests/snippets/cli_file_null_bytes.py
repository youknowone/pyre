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

    # `pymain_run_startup` enters the same tokenizer file boundary as the
    # ordinary script path.  A startup error is reported and the forced prompt
    # still runs, so feed it an explicit exit command.
    startup = os.path.join(directory, "startup_nul.py")
    with open(startup, "wb") as stream:
        stream.write(b"x = '\0' trailing bytes\n")
    env = os.environ.copy()
    env["PYTHONSTARTUP"] = startup
    result = subprocess.run(
        [sys.executable, "-i"],
        input=b"exit()\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result
    stderr = result.stderr.decode("utf-8", "replace")
    assert 'File "%s", line 1' % startup in stderr, stderr
    # `sys.stderr` ends a line with `os.linesep`, so compare the report by
    # lines: the offending source text is the line before the SyntaxError.
    lines = stderr.splitlines()
    reported = "SyntaxError: source code cannot contain null bytes"
    assert reported in lines, stderr
    assert lines[lines.index(reported) - 1] == "    x = '", stderr
    assert "\0" not in stderr, stderr

print("OK")
