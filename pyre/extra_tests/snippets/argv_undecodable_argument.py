"""A command-line argument with no UTF-8 spelling reaches `sys.argv` as itself.

`targetpypystandalone.py:76-80` builds `sys.argv` with `space.newfilename`,
which is `fsdecode(newbytes(s))`, so an argument carrying a byte the filesystem
encoding cannot spell arrives as the surrogate escape that re-encodes to that
byte — not rejected, not replaced. `sys.orig_argv` carries the same value.

The argument is passed to a child, so the test needs no such name on disk: the
filesystem never sees it, only `execve` does. Windows has no byte argv at all
and takes the wide command line, so this shape does not exist there.
"""

import os
import subprocess
import sys

if sys.platform == "win32":
    print("OK")
    raise SystemExit

UNDECODABLE = b"pyre_undecodable_\xff"
ESCAPED = os.fsdecode(UNDECODABLE)

# The escape is what the filesystem decode produces, and it round-trips.
assert ESCAPED.endswith("\udcff"), ascii(ESCAPED)
assert os.fsencode(ESCAPED) == UNDECODABLE, ascii(ESCAPED)

CHILD = r"""
import os, sys
assert sys.argv[1:] == [os.fsdecode(%r), "plain"], ascii(sys.argv)
assert os.fsencode(sys.argv[1]) == %r, ascii(sys.argv[1])
# `orig_argv` is the launcher's own line, so the argument appears there too,
# with the same escaping.
assert sys.argv[1] in sys.orig_argv, ascii(sys.orig_argv)
assert sys.orig_argv[-2:] == sys.argv[1:], ascii(sys.orig_argv)
print("child ok")
""" % (UNDECODABLE, UNDECODABLE)

result = subprocess.run(
    [sys.executable, "-c", CHILD, ESCAPED, "plain"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
assert result.returncode == 0, (result.returncode, result.stderr)
assert result.stdout == b"child ok\n", result.stdout

# A script run the same way answers with the argument in argv[1], and the
# script's own path stays argv[0].
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    script = os.path.join(tmp, "show_argv.py")
    with open(script, "w") as f:
        f.write(
            "import os, sys\n"
            "print(os.fsencode(sys.argv[0]) == os.fsencode(%r))\n" % script
            + "print(ascii(sys.argv[1]))\n"
        )
    result = subprocess.run(
        [sys.executable, script, ESCAPED],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert result.stdout == b"True\n" + ascii(ESCAPED).encode() + b"\n", result.stdout

print("OK")
# CPython-suite gap: no test invokes the executable with this exact undecodable
# argv byte. This generic process-boundary contract belongs in snippets.
