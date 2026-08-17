# CPython-suite gap: no suite test rebinds a descriptor under a hot JIT loop
# that folded it, and none installs a classmethod subclass as `__getattr__`.
# parity-tests reason: these are pyre JIT descriptor-binding regressions.

# The inline of a type's `__getattr__` hook resolves the descriptor spelling at
# record time — a plain function is called with the receiver, a `classmethod` or
# `staticmethod` is unwrapped to the callable inside it.  Two things that
# resolution must not assume, both of which produced a stale answer from the
# compiled trace while the interpreter answered correctly:
#
#  * `get_and_call_function` (`descroperation.py:169-187`) takes the descriptor
#    shortcut only for the EXACT type and routes everything else through
#    `space.get`.  A `classmethod` subclass overriding `__get__` binds through
#    that override, so unwrapping its `w_function` calls the wrong callable.
#
#  * `function.py:673`/`:720` `_immutable_fields_ = ['w_function?']`.  The `?`
#    registers the invalidation an assignment owes, so re-initialising an
#    installed wrapper has to be observed.  It changes no type's version tag,
#    which is the only pin the fold holds over the descriptor.
#
# The rebinds happen INSIDE each loop, because a read after the loop is
# interpreted and would not consult what the trace baked.  Every hook body
# returns a constant so the fold stands rather than aborting on a residual.

N = 12000
SWITCH = N // 2


class Subclassed(classmethod):
    def __get__(self, obj, objtype=None):
        return lambda name: 2


def first(cls_or_name, name=None):
    return 1


def second(cls_or_name, name=None):
    return 2


def exact_type_is_required():
    class Owner:
        __getattr__ = Subclassed(first)

    owner = Owner()
    last = None
    for _ in range(N):
        last = owner.miss
    return last


def rebind(wrapper):
    class Owner:
        __getattr__ = wrapper(first)

    owner = Owner()
    seen = []
    for index in range(N):
        value = owner.miss
        if index == SWITCH:
            Owner.__dict__['__getattr__'].__init__(second)
        elif index in (SWITCH - 1, N - 1):
            seen.append(value)
    return seen


def rebind_plain():
    class Owner:
        __getattr__ = first

    owner = Owner()
    seen = []
    for index in range(N):
        value = owner.miss
        if index == SWITCH:
            Owner.__getattr__ = second
        elif index in (SWITCH - 1, N - 1):
            seen.append(value)
    return seen


assert exact_type_is_required() == 2, 'overridden __get__ was bypassed'
assert rebind(classmethod) == [1, 2], 'classmethod w_function stayed baked'
assert rebind(staticmethod) == [1, 2], 'staticmethod w_function stayed baked'
assert rebind_plain() == [1, 2], 'rebound plain hook stayed baked'

print("OK")
