# pyre-check: gate=1
# A class carries the function attributes only when its MRO defines them.
# Answering None instead breaks `getattr(fn, "__func__", fn)`, the
# staticmethod/classmethod unwrap, which then carries None forward.
FUNCTION_ATTRS = (
    '__code__',
    '__func__',
    '__self__',
    '__globals__',
    '__closure__',
    '__defaults__',
    '__kwdefaults__',
    '__wrapped__',
)


class Plain:
    pass


class WithInit:
    def __init__(self):
        pass


for owner in (Plain, WithInit, type, object, str, int):
    for name in FUNCTION_ATTRS:
        assert not hasattr(owner, name), (owner, name)
        marker = object()
        assert getattr(owner, name, marker) is marker, (owner, name)
        try:
            getattr(owner, name)
        except AttributeError:
            pass
        else:
            raise AssertionError((owner, name))

# The unwrap idiom must hand back the class untouched.
assert getattr(Plain, '__func__', Plain) is Plain

# A class that really defines one still resolves it, through the descriptor
# protocol rather than a raw dict read.
def helper():
    return 1


class Declares:
    __func__ = staticmethod(helper)


assert Declares.__func__ is helper

# An instance method's own attributes are unaffected.
class Owner:
    def method(self):
        return self


instance = Owner()
assert instance.method.__func__ is Owner.method
assert instance.method.__self__ is instance
assert Owner.method.__code__.co_name == 'method'
