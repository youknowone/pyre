# pyre-check: no-cpython
# pyre-check: skip-backends=wasm
# The wasm guest has no libc calendar formatter; this fixture verifies the
# native Unix/Windows result allocation shared by their strftime branches.
import gc
import time


value = time.strftime(
    "%Y-%m-%d %H:%M:%S",
    (2020, 2, 3, 4, 5, 6, 0, 34, -1),
)

assert value == "2020-02-03 04:05:06"
assert any(obj is value for obj in gc.get_objects())

print("time.strftime result is collectable")
