"""A path entry the filesystem encoding cannot spell raises, it is not skipped.

`os.fsencode` refuses a surrogate outside the escape range (U+DC80..U+DCFF), so
an entry like `'\\ud800'` on `sys.path` or in a package's `__path__` names no
file and cannot be passed over: the import reports the encoding failure rather
than searching on and blaming a missing module.

No filesystem support is needed to reach this, unlike the sibling
`os_undecodable_name_boundaries.py` — the entry never gets as far as a syscall.
"""

import os
import sys
import tempfile

UNENCODABLE = "\ud800"

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
    pass
except ImportError as exc:
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
            pass
        except ImportError as exc:
            raise AssertionError("__path__ entry was skipped: %r" % (exc,))
        else:
            raise AssertionError("imported a submodule that does not exist")
    finally:
        sys.path.pop(0)
        sys.modules.pop("pyre_pathtest_pkg", None)

print("OK")
