import gc


# intobject.py:_as_subint allocates both W_IntObject and W_LongObject user
# subtypes through space.allocate_instance.  They therefore have ordinary GC
# lifetime, participate in the finalizer queue, and normalize back to an exact
# base int through the inherited conversion methods.
finalized = []


class FinalizingInt(int):
    def __del__(self):
        finalized.append(self.real)


for value in (42, 1 << 100):
    obj = FinalizingInt(value)
    assert type(int(obj)) is int
    assert type(obj.__int__()) is int
    assert type(obj.__index__()) is int
    del obj
    gc.collect()

assert finalized == [42, 1 << 100]

print("OK")
