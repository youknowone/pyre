# pyre-check: no-cpython
# pyre-check: skip-backends=wasm
import gc
import time


asctime_value = time.asctime(
    (2020, 2, 3, 4, 5, 6, 0, 34, -1),
)

assert asctime_value == "Mon Feb  3 04:05:06 2020"
assert any(obj is asctime_value for obj in gc.get_objects())

# The wasm guest does not register the time module. Native Unix and Windows
# share this localtime -> _asctime path.
ctime_value = time.ctime(0)
assert any(obj is ctime_value for obj in gc.get_objects())

print("time asctime results are collectable")
