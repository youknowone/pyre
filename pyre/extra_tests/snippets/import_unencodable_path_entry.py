# pyre-check: platforms=linux,darwin
# CPython-suite gap: import tests omit this unencodable sys.path entry.
# It is a generic import/filesystem boundary, so it belongs in snippets.
# `os.fsencode` rejects a surrogate outside U+DC80..U+DCFF only under the
# `surrogateescape` handler; Windows encodes with `surrogatepass` and accepts
# it, so the reference never reaches the import at all.
"""A path entry the filesystem encoding cannot spell raises, it is not skipped.

`os.fsencode` refuses a surrogate outside the escape range (U+DC80..U+DCFF), so
an entry like `'\\ud800'` on `sys.path` or in a package's `__path__` names no
file and cannot be passed over: the import reports the encoding failure rather
than searching on and blaming a missing module.

Windows encodes with `surrogatepass` (PEP 529) instead, which gives that same
entry a spelling — its own three UTF-8 bytes — so there the entry is searched
like any other and names a file that is simply not there.

No filesystem support is needed to reach this, unlike the sibling
`os_undecodable_name_boundaries.py` — the entry never gets as far as a syscall.
"""

import os
import sys
import tempfile

UNENCODABLE = "\ud800"
SPELLED = sys.platform == "win32"

if SPELLED:
    assert os.fsencode(UNENCODABLE) == b"\xed\xa0\x80", ascii(os.fsencode(UNENCODABLE))
else:
    try:
        os.fsencode(UNENCODABLE)
    except UnicodeEncodeError:
        pass
    else:
        raise AssertionError("fsencode accepted a lone surrogate")

# sys.path
sys.path.insert(0, UNENCODABLE)
try:
    import pyre_absent_module_for_path_test
except UnicodeEncodeError:
    if SPELLED:
        raise AssertionError("the entry has a spelling here and must be searched")
except ImportError as exc:
    if not SPELLED:
        raise AssertionError("entry was skipped instead of reported: %r" % (exc,))
else:
    raise AssertionError("imported a module that does not exist")
finally:
    sys.path.pop(0)

# A package's __path__ takes the same treatment.
with tempfile.TemporaryDirectory() as d:
    pkg = os.path.join(d, "pyre_pathtest_pkg")
    os.mkdir(pkg)
    with open(os.path.join(pkg, "__init__.py"), "w") as fp:
        fp.write("")
    sys.path.insert(0, d)
    try:
        import pyre_pathtest_pkg

        pyre_pathtest_pkg.__path__.insert(0, UNENCODABLE)
        try:
            import pyre_pathtest_pkg.absent
        except UnicodeEncodeError:
            if SPELLED:
                raise AssertionError("the entry has a spelling here and must be searched")
        except ImportError as exc:
            if not SPELLED:
                raise AssertionError("__path__ entry was skipped: %r" % (exc,))
        else:
            raise AssertionError("imported a submodule that does not exist")
    finally:
        sys.path.pop(0)
        sys.modules.pop("pyre_pathtest_pkg", None)

print("OK")
