# cpyext-expect: cpyext-abi3-finder-ok

# End-to-end check for single-phase extension loading.
#
# The fixture is compiled against pyre's own 3.14 ABI header, discovered by
# the normal import path, loaded through `PyInit_*`, and created by the C call
# back into the executable's exported `PyModule_Create2`.

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'lib_pypy'))
import _imp
import _pypy_abi3_tags
assert '.abi3.so' in _imp.extension_suffixes(), _imp.extension_suffixes()
assert any(isinstance(f, _pypy_abi3_tags._Abi3TagsFinder) for f in sys.meta_path)
print('cpyext-abi3-finder-ok')
