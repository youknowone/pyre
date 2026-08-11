# pyre-check: no-cpython

import gc
import re


pattern = re.compile("a+")
rendered = type(pattern).__repr__(pattern)

assert rendered == "re.compile('a+')"
assert any(obj is rendered for obj in gc.get_objects())

print("sre pattern repr is collectable")
