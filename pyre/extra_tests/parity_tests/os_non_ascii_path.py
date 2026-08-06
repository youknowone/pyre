"""A filesystem name outside ASCII survives every path call.

Windows names files by their wide (UTF-16) spelling; the narrow entry points
the C runtime also offers re-encode the name through the process's ANSI code
page, which for most of these characters has no spelling at all and answers
`ERROR_NO_UNICODE_TRANSLATION`.  So each path call takes the `W` form, and the
name a directory listing gives back is the name that was created.
"""

import os
import tempfile

# A name from three scripts, none of which a single legacy code page holds.
NAME = "한글-日本語-ελλ"

base = tempfile.mkdtemp(prefix="pyre_nonascii_")
directory = os.path.join(base, NAME + "_dir")
plain = os.path.join(base, NAME + ".txt")
renamed = os.path.join(base, NAME + "_renamed.txt")

os.mkdir(directory)
assert os.path.isdir(directory), directory
assert os.stat(directory).st_size >= 0

fd = os.open(plain, os.O_CREAT | os.O_WRONLY, 0o666)
assert os.write(fd, b"hello") == 5
os.close(fd)

assert os.stat(plain).st_size == 5, os.stat(plain).st_size
with open(plain, "rb") as fp:
    assert fp.read() == b"hello"

# The listing spells the names back exactly, both as `str` and as `bytes`.
names = sorted(os.listdir(base))
assert names == sorted([NAME + "_dir", NAME + ".txt"]), names
assert sorted(os.scandir(base), key=lambda e: e.name)[0].name == names[0]
byte_names = sorted(os.listdir(os.fsencode(base)))
assert byte_names == sorted(os.fsencode(n) for n in names), byte_names

os.rename(plain, renamed)
assert os.path.exists(renamed) and not os.path.exists(plain)

# And each of them names the same file to a call that removes it.
os.remove(renamed)
os.rmdir(directory)
assert os.listdir(base) == []
os.rmdir(base)
print("OK")
