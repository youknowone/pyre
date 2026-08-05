"""A filename with no UTF-8 spelling survives every boundary that reads it back.

`os.listdir`/`os.scandir` decode `readdir`'s `d_name` with `surrogateescape`
(`interp_posix.py:1121 space.newfilename`), so a name like `b"bad_\xff"` reaches
Python as a `str` holding U+DCFF.  Every consumer that then re-reads that `str`
has to take the filesystem encoding back out of it — a `DirEntry` repr, an
`__import__` driven by `sys.path`, and the `fork_exec` argv boundary all read
the same object — rather than demanding a UTF-8 view it cannot have.

The name is only creatable where the kernel takes arbitrary bytes: APFS/HFS+
answer EILSEQ, so the whole file is skipped where the plant fails.
"""

import os
import subprocess
import sys
import tempfile

if sys.platform == "win32":
    print("OK")
    raise SystemExit

RAW = b"pyre_undecodable_\xff"
NAME = os.fsdecode(RAW)

with tempfile.TemporaryDirectory() as d:
    try:
        os.mkdir(os.path.join(os.fsencode(d), RAW))
    except (OSError, ValueError):
        # The filesystem refuses a name that is not valid UTF-8 (APFS/HFS+).
        print("OK")
        raise SystemExit

    # `os.listdir` hands the name back with its escape intact.
    assert os.listdir(d) == [NAME], ascii(os.listdir(d))

    # `DirEntry.__repr__` is `"<DirEntry %R>"` (posixmodule.c), so the name is
    # rendered by its own repr and keeps the escape.
    entries = list(os.scandir(d))
    assert len(entries) == 1, entries
    assert repr(entries[0]) == "<DirEntry %r>" % NAME, repr(entries[0])

    # In bytes mode the name is `bytes`, and the repr says so.
    bentries = list(os.scandir(os.fsencode(d)))
    assert bentries[0].name == RAW, bentries[0].name
    assert repr(bentries[0]) == "<DirEntry %r>" % RAW, repr(bentries[0])

    # An import driven by such a directory has to encode it back to the bytes
    # that name it on disk.  The package below is planted inside the
    # undecodable directory, so it is importable only if the whole path
    # round-trips.
    pkg = os.path.join(os.fsencode(d), RAW, b"pyre_undecodable_mod.py")
    with open(pkg, "wb") as fp:
        fp.write(b"VALUE = 41\n")
    sys.path.insert(0, os.path.join(d, NAME))
    try:
        import pyre_undecodable_mod

        assert pyre_undecodable_mod.VALUE == 41
    finally:
        sys.path.pop(0)
        sys.modules.pop("pyre_undecodable_mod", None)

    # The exec boundary takes the same encoding: the name reaches the OS as
    # bytes rather than raising, and a program by that name does not exist.
    try:
        subprocess.run([os.path.join(d, NAME, NAME)], capture_output=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("an absent program was found")

print("OK")
