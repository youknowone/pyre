# CPython-suite gap: no suite test installs a classmethod subclass as `__getattr__`.
# parity-tests reason: this guards the exact-descriptor check in pyre's JIT hook fold.

# `get_and_call_function` takes its descriptor shortcut only for the exact
# classmethod/staticmethod type. A subclass overriding `__get__` must bind
# through that override. Wrapper invalidation is already owned by
# `bench/synth/wrapper_function_invalidation.py`.

N = 12000
class Subclassed(classmethod):
    def __get__(self, obj, objtype=None):
        return lambda name: 2


def first(cls_or_name, name=None):
    return 1


def exact_type_is_required():
    class Owner:
        __getattr__ = Subclassed(first)

    owner = Owner()
    last = None
    for _ in range(N):
        last = owner.miss
    return last
assert exact_type_is_required() == 2, 'overridden __get__ was bypassed'
print("OK")
