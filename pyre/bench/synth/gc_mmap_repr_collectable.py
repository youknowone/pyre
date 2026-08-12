# pyre-check: no-cpython
# pyre-check: skip-backends=wasm
import gc
import mmap


# pyre's mmap implementation is currently Unix-only.  The module still
# imports on Windows so callers can feature-detect it, but it does not expose
# mmap.mmap there.
if not hasattr(mmap, "mmap"):
    print("mmap repr results are collectable")
    raise SystemExit

mapping = mmap.mmap(-1, 1)
live_direct = mmap.mmap.__repr__(mapping)
live_ordinary = repr(mapping)

assert live_direct == live_ordinary
assert "closed=False" in live_direct
assert any(obj is live_direct for obj in gc.get_objects())
assert any(obj is live_ordinary for obj in gc.get_objects())

mapping.close()
closed_direct = mmap.mmap.__repr__(mapping)
closed_ordinary = repr(mapping)

assert closed_direct == closed_ordinary
assert closed_direct == "<mmap.mmap closed=True>"
assert any(obj is closed_direct for obj in gc.get_objects())
assert any(obj is closed_ordinary for obj in gc.get_objects())

print("mmap repr results are collectable")
