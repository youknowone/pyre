# pyre-check: platforms=linux,darwin
# The asserted spelling is the `posix` module's; on Windows `os` re-exports
# these from `nt`, so every qualified name below differs in the reference too.
"""`os.DirEntry` / `posix.ScandirIterator` are produced only by `os.scandir`.

`interp_scandir.py:468-487` gives `W_DirEntry.typedef` no `__new__` and sets
`acceptable_as_base_class = False` (:487); `:172-180` does the same for
`W_ScandirIterator.typedef`.  `:463-465 descr_reduce_ex` refuses to pickle an
entry, spelling the type with `%T` (`error.py:592-593` -> the qualified typedef
name `'posix.DirEntry'`, `:469`).  The module holding these types is the one
`os` is implemented by, which Windows spells `nt`.

Pickle protocols 0 and 1 are deliberately NOT asserted: they never reach
`reduce_newobj`, they land in `copyreg._reduce_ex`, and that leg already
diverges for every static type.  It is a separate, pre-existing defect.
"""

import copy
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = "nt" if sys.platform == "win32" else "posix"

assert os.DirEntry.__module__ == MODULE, os.DirEntry.__module__
assert os.DirEntry.__name__ == "DirEntry", os.DirEntry.__name__
assert os.DirEntry.__qualname__ == "DirEntry", os.DirEntry.__qualname__
assert repr(os.DirEntry) == "<class '%s.DirEntry'>" % MODULE, repr(os.DirEntry)

try:
    os.DirEntry()
except TypeError as exc:
    assert str(exc) == "cannot create '%s.DirEntry' instances" % MODULE, str(exc)
else:
    raise AssertionError("os.DirEntry() must not construct an entry")

try:
    class _Sub(os.DirEntry):
        pass
except TypeError as exc:
    expected = "type '%s.DirEntry' is not an acceptable base type" % MODULE
    assert str(exc) == expected, str(exc)
else:
    raise AssertionError("os.DirEntry must not be an acceptable base type")

it = os.scandir(HERE)
scandir_iterator = type(it)
assert scandir_iterator.__module__ == MODULE, scandir_iterator.__module__
assert scandir_iterator.__name__ == "ScandirIterator", scandir_iterator.__name__

try:
    scandir_iterator()
except TypeError as exc:
    assert str(exc) == "cannot create '%s.ScandirIterator' instances" % MODULE, str(exc)
else:
    raise AssertionError("ScandirIterator() must not construct an iterator")

try:
    class _SubIter(scandir_iterator):
        pass
except TypeError as exc:
    expected = "type '%s.ScandirIterator' is not an acceptable base type" % MODULE
    assert str(exc) == expected, str(exc)
else:
    raise AssertionError("ScandirIterator must not be an acceptable base type")

entries = list(it)
it.close()
assert entries
entry = entries[0]
assert isinstance(entry, os.DirEntry)
assert isinstance(entry.name, str)
assert entry.path == os.path.join(HERE, entry.name), (entry.path, entry.name)

for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
    try:
        pickle.dumps(entry, protocol)
    except TypeError as exc:
        assert str(exc) == "cannot pickle '%s.DirEntry' object" % MODULE, (protocol, str(exc))
    else:
        raise AssertionError("a DirEntry must refuse to be pickled")

for clone in (copy.copy, copy.deepcopy):
    try:
        clone(entry)
    except TypeError as exc:
        assert str(exc) == "cannot pickle '%s.DirEntry' object" % MODULE, (clone, str(exc))
    else:
        raise AssertionError("a DirEntry must refuse to be copied")

print("OK")
