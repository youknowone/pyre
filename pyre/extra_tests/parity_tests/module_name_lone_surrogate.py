# CPython-suite gap: test_module never names a module with a lone surrogate.
# parity-tests reason: PyPy keeps the name object, so no encoding step can reject it.

"""A module name carrying a lone surrogate survives construction and lookup.

The import machinery reaches such a name whenever a filename was decoded with
surrogateescape, which is how `test_import.test_unencodable_filename` imports
`TESTFN_UNENCODABLE`.  Reading it as UTF-8 anywhere on the way in aborts the
interpreter rather than raising.
"""

import types

name = "mod-\udcff"

module = types.ModuleType(name)
assert module.__name__ is name
assert module.__dict__["__name__"] is name

# `module.__init__` re-seeds the name on an already-built module, the path
# `module.__new__` leaves for the import machinery to fill in.
reseeded = types.ModuleType("anonymous")
reseeded.__init__(name)
assert reseeded.__name__ is name

# `repr` formats the name rather than storing it, so it takes its own route
# out — through `repr` of the name, which escapes the surrogate.
assert repr(name)[1:-1] in repr(module)

# A surrogate name is an ordinary dict key, so the module resolves under it.
registry = {module.__name__: module}
assert registry[name] is module

print("OK")
