"""Python 3.14 `_imp.is_builtin` reports inittab membership, not import state."""

import _imp


assert _imp.is_builtin("time") == 1
import time
assert _imp.is_builtin("time") == 1
assert _imp.is_builtin("os") == 0
assert _imp.is_builtin("not existing module") == 0

print("OK")
