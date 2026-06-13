# C1: stdlib copyreg.py is visible and pickle imports.
import copyreg
assert hasattr(copyreg, "_extension_registry"), "copyreg._extension_registry missing"
assert hasattr(copyreg, "_inverted_registry"), "copyreg._inverted_registry missing"
assert hasattr(copyreg, "_extension_cache"), "copyreg._extension_cache missing"
assert hasattr(copyreg, "__newobj__"), "copyreg.__newobj__ missing"
assert hasattr(copyreg, "_reconstructor"), "copyreg._reconstructor missing"
assert callable(copyreg._reduce_ex), "copyreg._reduce_ex missing"

import pickle
assert hasattr(pickle, "dumps") and hasattr(pickle, "loads")
print("pickle_import OK")
