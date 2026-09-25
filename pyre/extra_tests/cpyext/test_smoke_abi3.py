# cpyext-fixture: cpyext_smoke .abi3.so
# cpyext-expect: cpyext-abi3-ok

# End-to-end check for single-phase extension loading.
#
# The fixture is compiled against pyre's own 3.14 ABI header, discovered by
# the normal import path, loaded through `PyInit_*`, and created by the C call
# back into the executable's exported `PyModule_Create2`.

import _imp
import cpyext_smoke
assert '.abi3.so' in _imp.extension_suffixes()
assert cpyext_smoke.__name__ == 'cpyext_smoke'
assert cpyext_smoke.__file__.endswith('.abi3.so'), cpyext_smoke.__file__
print('cpyext-abi3-ok')
