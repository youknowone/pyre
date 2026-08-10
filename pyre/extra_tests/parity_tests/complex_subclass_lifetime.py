# CPython-suite gap: complex-subclass tests omit user-finalizer lifetime.
# parity-tests reason: this guards PyPy-style allocation and moving-GC ownership.

import gc


# complexobject.py:descr__new__ allocates strict subtypes through
# space.allocate_instance.  They therefore have ordinary GC lifetime, enter
# the finalizer queue, and normalize to an exact base complex through the
# inherited conversion method.
finalized = []


class FinalizingComplex(complex):
    def __del__(self):
        finalized.append((self.real, self.imag))


obj = FinalizingComplex(1.25, -2.5)
assert type(complex(obj)) is complex
assert type(obj.__complex__()) is complex
del obj
gc.collect()

assert finalized == [(1.25, -2.5)]

print("OK")
