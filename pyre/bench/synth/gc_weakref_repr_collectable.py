# pyre-check: no-cpython
import gc
import weakref


class Referent:
    pass


class CallableReferent:
    def __call__(self):
        pass


referent = Referent()
callable_referent = CallableReferent()
reference = weakref.ref(referent)
proxy = weakref.proxy(referent)
callable_proxy = weakref.proxy(callable_referent)

live_results = (
    weakref.ReferenceType.__repr__(reference),
    repr(reference),
    weakref.ProxyType.__repr__(proxy),
    repr(proxy),
    weakref.CallableProxyType.__repr__(callable_proxy),
    repr(callable_proxy),
)

for result in live_results:
    assert any(obj is result for obj in gc.get_objects())

del referent
del callable_referent
gc.collect()

dead_results = (
    weakref.ReferenceType.__repr__(reference),
    repr(reference),
    weakref.ProxyType.__repr__(proxy),
    repr(proxy),
    weakref.CallableProxyType.__repr__(callable_proxy),
    repr(callable_proxy),
)

for result in dead_results:
    assert "; dead>" in result
    assert any(obj is result for obj in gc.get_objects())

print("weakref repr results are collectable")
