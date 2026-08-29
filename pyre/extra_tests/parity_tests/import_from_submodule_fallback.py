# CPython-suite gap: `test_import` covers `from pkg import sub` through the
# importer, but nothing there reaches IMPORT_FROM with the parent attribute
# missing while `sys.modules` still holds the submodule -- the one state in
# which the opcode's own fallback is what answers.
# parity-tests reason: that fallback is a `sys.modules` LOOKUP. It must not run
# an import and must not bind the result back onto the parent, because
# `__import__`'s `_handle_fromlist` already did whatever binding is owed and a
# second binding is observable as an attribute the program deleted coming back.

import json
import os
import shutil
import sys
import tempfile
import types

root = tempfile.mkdtemp(prefix="pyre_import_from_")
os.mkdir(os.path.join(root, "pfpkg"))
os.mkdir(os.path.join(root, "pfpkg", "sub"))
with open(os.path.join(root, "pfpkg", "__init__.py"), "w") as f:
    f.write("NAME = 'pfpkg'\n")
with open(os.path.join(root, "pfpkg", "leaf.py"), "w") as f:
    f.write("VALUE = 7\n")
with open(os.path.join(root, "pfpkg", "sub", "__init__.py"), "w") as f:
    f.write("")
with open(os.path.join(root, "pfpkg", "sub", "deep.py"), "w") as f:
    f.write("DEEP = 11\n")
with open(os.path.join(root, "pfpkg", "rel.py"), "w") as f:
    f.write("from . import leaf\nfrom .sub import deep\nOUT = (leaf.VALUE, deep.DEEP)\n")
sys.path.insert(0, root)

# `_handle_fromlist` is what imports a submodule named in a fromlist, so these
# succeed on the plain-getattr path: by the time IMPORT_FROM runs, the importer
# has already bound the name on the parent.
assert "pfpkg.leaf" not in sys.modules
from pfpkg import leaf
assert leaf.VALUE == 7
assert sys.modules["pfpkg.leaf"] is leaf
assert sys.modules["pfpkg"].leaf is leaf

from pfpkg.sub import deep
assert deep.DEEP == 11

from pfpkg import rel
assert rel.OUT == (7, 11)

# The fallback proper: delete the attribute the importer bound. The submodule
# stays in `sys.modules`, so `_handle_fromlist` re-imports nothing and rebinds
# nothing, and IMPORT_FROM answers from `sys.modules` -- WITHOUT putting the
# attribute back.
assert hasattr(json, "decoder")
delattr(json, "decoder")
from json import decoder
assert decoder is sys.modules["json.decoder"]
assert not hasattr(json, "decoder"), "IMPORT_FROM rebound the parent attribute"

# Repeating it keeps taking the fallback, which is only true because the first
# one stored nothing.
from json import decoder as decoder2
assert decoder2 is decoder
assert not hasattr(json, "decoder")

# Issue #17636 -- the `from` target need not be a module. `__name__` is read off
# the object, so a stand-in with a registered `<__name__>.<name>` resolves.
class Stand:
    __slots__ = ("__name__",)


stand = Stand()
stand.__name__ = "pfstand"
piece = types.ModuleType("pfstand.piece")
sys.modules["pfstand"] = stand
sys.modules["pfstand.piece"] = piece
from pfstand import piece as got_piece
assert got_piece is piece
assert not hasattr(stand, "piece")

# A module `__getattr__` answers first, so the fallback is never consulted.
holder = types.ModuleType("pfdyn")


class DynModule(types.ModuleType):
    def __getattr__(self, key):
        if key == "dyn":
            return "dynamic"
        raise AttributeError(key)


holder.__class__ = DynModule
sys.modules["pfdyn"] = holder
from pfdyn import dyn
assert dyn == "dynamic"

# Neither an attribute nor a registered submodule: ImportError naming both the
# missing name and the package, with the package's `__file__` as the location.
try:
    from pfpkg import nope
except ImportError as exc:
    assert exc.name == "pfpkg", exc.name
    message = str(exc)
    assert "cannot import name 'nope' from 'pfpkg'" in message, message
    assert os.path.join("pfpkg", "__init__.py") in message, message
else:
    raise AssertionError("expected ImportError")

sys.path.remove(root)
shutil.rmtree(root, ignore_errors=True)
print("OK")
