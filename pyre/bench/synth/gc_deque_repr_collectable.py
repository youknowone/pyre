# pyre-check: no-cpython

import gc
from collections import deque


rendered = deque.__repr__(deque([1, 2, 3], maxlen=4))

assert rendered == "deque([1, 2, 3], maxlen=4)"
assert any(obj is rendered for obj in gc.get_objects())

print("deque repr is collectable")
