# CPython-suite gap: descriptor-call tests do not force collection while binding.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.

"""A custom ``__call__`` descriptor must not invalidate call arguments."""

import gc


class CallDescriptor:
    def __get__(self, instance, owner):
        # Descriptor binding is an arbitrary Python callback.  Force a moving
        # collection before the call dispatcher consumes its original args.
        garbage = [[index] for index in range(2000)]
        assert len(garbage) == 2000
        gc.collect()

        def bound(*args, **kwargs):
            return instance, args, kwargs

        return bound


class Callable:
    __call__ = CallDescriptor()


class Payload:
    pass


callable_object = Callable()
for _ in range(20):
    positional = Payload()
    keyword = Payload()

    receiver, args, kwargs = callable_object(positional)
    assert receiver is callable_object
    assert args == (positional,)
    assert kwargs == {}

    receiver, args, kwargs = callable_object(positional, keyword=keyword)
    assert receiver is callable_object
    assert args == (positional,)
    assert kwargs == {"keyword": keyword}

print("OK")
