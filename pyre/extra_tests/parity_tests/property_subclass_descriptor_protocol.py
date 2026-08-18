# CPython-suite gap: the suite's property tests subclass `property` to add
# methods, never to override `__get__` / `__set__` / `__delete__`.
# parity-tests reason: this is a pyre descriptor-dispatch divergence, shared by
# the interpreter and the JIT fold over it.

# `get_and_call_function` (`descroperation.py:169-176`) takes a descriptor
# shortcut only for the EXACT type — "isinstance(typ, Function) would not be
# correct here" — and routes everything else through `space.get`, i.e.
# `type(w_descr).__get__` off the MRO.  A `property` subclass keeps the base
# layout and retags only its class word, so a layout test admits it and calls
# the wrapped `fget` in place of the override.
#
# The hot loop is here because the JIT's property fold applies the same
# resolution: it must decline for a subclass rather than bake the wrapped
# accessor.

N = 12000


class Overriding(property):
    def __get__(self, obj, objtype=None):
        return 'override-get'

    def __set__(self, obj, value):
        obj.recorded = 'override-set'

    def __delete__(self, obj):
        obj.recorded = 'override-del'


def base_getter(self):
    return 'wrapped-get'


def base_setter(self, value):
    self.recorded = 'wrapped-set'


def base_deleter(self):
    self.recorded = 'wrapped-del'


class WithOverride:
    recorded = None
    x = Overriding(base_getter, base_setter, base_deleter)


class Plain:
    # A subclass that overrides nothing still reaches `property`'s own
    # `__get__` through the MRO.
    recorded = None
    x = type('Inert', (property,), {})(base_getter, base_setter, base_deleter)


def overridden_accessors_run():
    obj = WithOverride()
    seen = set()
    for _ in range(N):
        seen.add(obj.x)
    assert seen == {'override-get'}, 'overridden __get__ was bypassed: %r' % (seen,)

    for _ in range(N):
        obj.x = 1
    assert obj.recorded == 'override-set', (
        'overridden __set__ was bypassed: %r' % (obj.recorded,)
    )

    del obj.x
    assert obj.recorded == 'override-del', (
        'overridden __delete__ was bypassed: %r' % (obj.recorded,)
    )


def inert_subclass_still_works():
    obj = Plain()
    seen = set()
    for _ in range(N):
        seen.add(obj.x)
    assert seen == {'wrapped-get'}, 'inert subclass lost its getter: %r' % (seen,)

    obj.x = 1
    assert obj.recorded == 'wrapped-set', (
        'inert subclass lost its setter: %r' % (obj.recorded,)
    )

    del obj.x
    assert obj.recorded == 'wrapped-del', (
        'inert subclass lost its deleter: %r' % (obj.recorded,)
    )


overridden_accessors_run()
inert_subclass_still_works()
print("OK")
