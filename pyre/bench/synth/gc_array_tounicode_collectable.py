# pyre-check: no-cpython

import array
import gc


rendered = array.array("u", "abc").tounicode()

assert rendered == "abc"
assert any(obj is rendered for obj in gc.get_objects())

print("array tounicode is collectable")
