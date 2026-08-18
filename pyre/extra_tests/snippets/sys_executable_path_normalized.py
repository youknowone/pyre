# pyre-check: gate=1
import os
import subprocess
import sys

# `initpath.py:find_executable` applies `abspath`, whose second half is
# `normpath` -- joining the cwd alone would carry the caller's spelling of
# `.` into `sys.executable` and every prefix derived from it, and `site.py`
# compares those prefixes against the directory it finds itself in.
directory, name = os.path.split(sys.executable)
parent, leaf = os.path.split(directory)

SHOW = 'import sys; print(sys.executable)'

spellings = [
    # Relative, resolved against the cwd.
    (os.path.join(os.curdir, name), directory),
    # Already absolute, so only the collapsing is left to do.
    (os.path.join(directory, os.curdir, name), parent),
    (os.path.join(directory, os.pardir, leaf, name), parent),
]

for spelling, cwd in spellings:
    result = subprocess.run(
        [spelling, '-c', SHOW],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, (spelling, result.stderr)
    reported = result.stdout.decode().strip()

    assert os.path.isabs(reported), (spelling, reported)
    parts = reported.split(os.sep)
    assert os.curdir not in parts, (spelling, reported)
    assert os.pardir not in parts, (spelling, reported)
    assert os.path.samefile(reported, sys.executable), (spelling, reported)
