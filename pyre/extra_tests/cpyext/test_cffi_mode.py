# cpyext-fixture: cpyext_cffi_mode
# cpyext-expect: cpyext-cffi-mode-ok

# Which of its two forms a module cffi generated compiles to.

import sys
import cpyext_cffi_mode as m

saw_version, saw_version_num, init_prefix, hexversion = m.chosen()
# `_CFFI_` is set, so neither macro is: the module builds its `_cffi_exports`
# side and imports `_cffi_backend` from `PyInit_`.
assert (saw_version, saw_version_num) == (0, 0), (saw_version, saw_version_num)
assert init_prefix == 'PyInit_', init_prefix
# The marker decides that one question and nothing else.
assert hexversion == sys.hexversion, (hex(hexversion), hex(sys.hexversion))

print('cpyext-cffi-mode-ok')
