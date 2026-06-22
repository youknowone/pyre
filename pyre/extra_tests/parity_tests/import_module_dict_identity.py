"""Phase 6 parity test: imported module __dict__ identity through import variants.

PyPy `pypy/module/imp/interp_imp.py` returns the same module object
identity for repeated `import x` of the same name (via `sys.modules`
cache), and the module's `__dict__` is a single `W_ModuleDictObject`
identity throughout its lifetime.

Pinned contract:
  1. Two `import sys` statements return the same module object,
  2. The module's `__dict__` identity is stable across re-imports,
  3. `from x import a` and `import x; x.a` return the same value,
  4. `sys.modules['sys']` is the same module identity.
"""

import sys

# (1) Repeated import returns same object.
import sys as sys2
assert sys is sys2, "repeated import must return same module"


# (2) __dict__ identity stable across the re-import.
d1 = sys.__dict__
import sys as sys3
d2 = sys3.__dict__
assert d1 is d2, "module __dict__ identity must persist across re-imports"


# (3) `from x import a` vs `import x; x.a`.
from sys import version as version_a
import sys as sys4
version_b = sys4.version
assert version_a is version_b, (
    f"from-import and dotted access must yield same identity: "
    f"{version_a is version_b}"
)


# (4) sys.modules['sys'] is sys itself.
assert sys.modules["sys"] is sys


# (5) math is also a stable identity.
import math
assert math.__dict__["__name__"] == "math"
import math as math2
assert math is math2
assert math.pi is math2.pi


print("OK")
