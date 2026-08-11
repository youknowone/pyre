# pyre-check: no-cpython

import gc
import sys


rendered = repr(sys.version_info)

assert rendered.startswith("sys.version_info(major=3, minor=")
assert any(obj is rendered for obj in gc.get_objects())

print("structseq repr is collectable")
