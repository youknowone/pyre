# pyre-check: no-cpython
import gc


view = memoryview(b"x")
live_direct = memoryview.__repr__(view)
live_ordinary = repr(view)

assert live_direct == live_ordinary
assert live_direct.startswith("<memory at 0x")
assert any(obj is live_direct for obj in gc.get_objects())
assert any(obj is live_ordinary for obj in gc.get_objects())

view.release()
released_direct = memoryview.__repr__(view)
released_ordinary = repr(view)

assert released_direct == released_ordinary
assert released_direct.startswith("<released memory at 0x")
assert any(obj is released_direct for obj in gc.get_objects())
assert any(obj is released_ordinary for obj in gc.get_objects())

print("memoryview repr results are collectable")
