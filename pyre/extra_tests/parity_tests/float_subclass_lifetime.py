# CPython-suite gap: float-subclass tests omit user-finalizer lifetime.
# parity-tests reason: this guards PyPy-style allocation and moving-GC ownership.

import gc


# floatobject.py:descr__new__ allocates a strict subtype through
# space.allocate_instance.  It therefore has ordinary GC lifetime, enters the
# finalizer queue, and normalizes to an exact base float through the inherited
# conversion methods.
finalized = []


class FinalizingFloat(float):
    def __del__(self):
        finalized.append(self.real)


obj = FinalizingFloat(1.25)
assert type(float(obj)) is float
assert type(obj.__float__()) is float
del obj
gc.collect()

assert finalized == [1.25]

print("OK")
