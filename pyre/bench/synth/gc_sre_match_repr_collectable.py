# pyre-check: no-cpython

import gc
import re


match = re.compile("a+").match("aaa")
rendered = type(match).__repr__(match)

assert rendered == "<re.Match object; span=(0, 3), match='aaa'>"
assert any(obj is rendered for obj in gc.get_objects())

print("sre match repr is collectable")
