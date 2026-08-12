# pyre-check: no-cpython
# pyre-check: skip-backends=wasm
import gc
import os


entry = next(os.scandir("."))
direct = os.DirEntry.__repr__(entry)
ordinary = repr(entry)

assert direct == ordinary
assert direct.startswith("<DirEntry ")
assert any(obj is direct for obj in gc.get_objects())
assert any(obj is ordinary for obj in gc.get_objects())

print("DirEntry repr results are collectable")
