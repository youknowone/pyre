# pyre-check: gate=1
import os
import subprocess
import sys
import tempfile

# `initpath.py:find_executable` applies `abspath`, whose second half is
# `normpath` -- joining the cwd alone would carry the caller's spelling of
# `.` into `sys.executable` and every prefix derived from it, and `site.py`
# compares those prefixes against the directory it finds itself in.
directory, name = os.path.split(sys.executable)
parent, leaf = os.path.split(directory)

SHOW = 'import sys; print(sys.executable)'


def report(spelling, cwd=None):
    """Spawn through `spelling` and hand back the child's `sys.executable`.

    The directory is entered rather than passed as `subprocess.run(cwd=...)`:
    that argument moves the child once it exists, and a relative program name
    is looked up before then, against this process's own directory.
    """
    previous = os.getcwd()
    if cwd is not None:
        os.chdir(cwd)
    try:
        result = subprocess.run(
            [spelling, '-c', SHOW],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        os.chdir(previous)
    assert result.returncode == 0, (spelling, result.stderr)
    return result.stdout.decode().strip()


spellings = [
    # Relative, resolved against the cwd.
    (os.path.join(os.curdir, name), directory),
    # Already absolute, so only the collapsing is left to do.
    (os.path.join(directory, os.curdir, name), parent),
    (os.path.join(directory, os.pardir, leaf, name), parent),
]

for spelling, cwd in spellings:
    reported = report(spelling, cwd)

    assert os.path.isabs(reported), (spelling, reported)
    parts = reported.split(os.sep)
    assert os.curdir not in parts, (spelling, reported)
    assert os.pardir not in parts, (spelling, reported)
    assert os.path.samefile(reported, sys.executable), (spelling, reported)

# The checks below compare the reported spelling character for character, and
# that is a contract of this implementation rather than a portable one: the
# reference interpreter's own `sys.executable` does not run through
# `posixpath.normpath` -- it answers a relative `../bin/python` unnormalized --
# and a darwin framework build answers with the image path the loader resolved,
# so neither the spelling nor its normalization survives there.  What is
# asserted is `initpath.py`'s division: `abspath` normalizes lexically and
# `resolvedirof` is the only half that follows a link.
if sys.implementation.name == 'pyre' and os.name == 'posix':
    # A symlinked executable answers with the link.  `samefile` cannot see the
    # difference -- it resolves both sides -- so this compares the strings.
    # Windows is left out because creating a symlink there needs a privilege
    # an ordinary account does not hold.
    with tempfile.TemporaryDirectory() as tmp:
        # The container is resolved up front so that the only unresolved link
        # left in the comparison is the one under test -- on darwin the
        # temporary directory itself sits under a symlinked `/var`, which the
        # child's own cwd reports resolved.
        tmp = os.path.realpath(tmp)
        link = os.path.join(tmp, 'pyre-link')
        os.symlink(sys.executable, link)

        assert report(link) == link, link
        assert report(os.path.join(os.curdir, 'pyre-link'), tmp) == link, link

    # `..` directly under the root names nothing, so it is dropped rather than
    # carried.  A path opening with exactly two slashes is the host's to
    # interpret and keeps both; three or more collapse to one.
    stripped = sys.executable.lstrip(os.sep)
    rooted = [
        (os.sep + os.pardir + sys.executable, sys.executable),
        (os.sep * 2 + stripped, os.sep * 2 + stripped),
        (os.sep * 3 + stripped, sys.executable),
    ]
    for spelling, expected in rooted:
        assert report(spelling) == expected, spelling
