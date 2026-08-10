import gc
import sys


class BrokenDel:
    def __del__(self):
        raise ValueError("del is broken")


seen = []
old_hook = sys.unraisablehook
sys.unraisablehook = seen.append
try:
    obj = BrokenDel()
    del_repr = repr(type(obj).__del__)
    del obj
    gc.collect()
finally:
    sys.unraisablehook = old_hook

assert len(seen) == 1
assert seen[0].err_msg == f"Exception ignored while calling deallocator {del_repr}"
assert seen[0].exc_type is ValueError
assert seen[0].object is None

print("OK")
