# `get_and_call_function`: a __getattr__ (or __getattribute__)
# defined as a classmethod or staticmethod must be bound through __get__ before
# being called, exactly like any other special method, so it receives the
# arguments the descriptor protocol gives it.
#
# The hook inlines against the version-tag and map pins that make the miss
# constant, instead of costing one opaque residual per access holding the
# whole `object_getattr_miss` walk plus a fresh frame.


class ClassmethodGetattr:
    @classmethod
    def __getattr__(cls, name):
        return 'cm:%s:%s' % (cls.__name__, name)


class StaticmethodGetattr:
    @staticmethod
    def __getattr__(name):
        return 'sm:%s' % name


class PlainGetattr:
    def __getattr__(self, name):
        return 'plain:%s' % name


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 10000


def main():
    # classmethod hook receives the class as its first bound argument
    print('classmethod', ClassmethodGetattr().nope)
    # staticmethod hook receives only the name
    print('staticmethod', StaticmethodGetattr().nope)
    # plain function hook stays bound to the instance
    print('plain', PlainGetattr().nope)

    cm = ClassmethodGetattr()
    sm = StaticmethodGetattr()
    plain = PlainGetattr()
    total = 0
    i = 0
    while i < N:
        if cm.nope == 'cm:ClassmethodGetattr:nope':
            total += 1
        if sm.nope == 'sm:nope':
            total += 1
        if plain.nope == 'plain:nope':
            total += 1
        i += 1
    print('loop', total)

    # The same fold is guarded by the receiver map and type version.  Stores
    # that invalidate either one must switch away from the recorded hook on
    # the very next iteration.
    class Mutable:
        def __getattr__(self, name):
            return "hook"

    obj = Mutable()
    seen = None
    for i in range(30000):
        seen = obj.later
        if i == 15000:
            obj.later = "instance"
    assert seen == "instance"

    class Replaced:
        @classmethod
        def __getattr__(cls, name):
            return "first"

    obj = Replaced()
    seen = None
    for i in range(30000):
        seen = obj.later
        if i == 15000:
            Replaced.__getattr__ = classmethod(lambda cls, name: "second")
    assert seen == "second"

    # A non-string key devolves mapdict.  Every devolved instance of the class
    # shares that map terminator, so the fold must consult the live dict rather
    # than treating a pinned map as proof that the name is absent.
    obj = Mutable()
    obj.__dict__[1] = "devolve"
    obj.present = "real"
    for _ in range(12000):
        assert obj.present == "real"
        assert obj.missing == "hook"
    for i in range(12000):
        value = obj.later
        if i == 6000:
            obj.later = "assigned"
    assert value == "assigned"


def type_shadow_case():
    class TypeShadow:
        @staticmethod
        def __getattr__(name):
            return "hook:" + name

    obj = TypeShadow()
    for i in range(12000):
        seen = obj.later
        if i == 6000:
            TypeShadow.later = "class"
    assert seen == "class"


def receiver_binding_case():
    class BoundByReceiver:
        @classmethod
        def __getattr__(cls, name):
            return cls.__name__ + ":" + name

    class Sub(BoundByReceiver):
        pass

    for _ in range(12000):
        assert BoundByReceiver().missing == "BoundByReceiver:missing"
        assert Sub().missing == "Sub:missing"


def raising_hook_case():
    class Raising:
        def __getattr__(self, name):
            raise AttributeError("no " + name)

    obj = Raising()
    misses = 0
    for _ in range(12000):
        try:
            obj.missing
        except AttributeError as exc:
            assert str(exc) == "no missing"
            misses += 1
    assert misses == 12000


def self_installing_hook_case():
    class Installing:
        def __getattr__(self, name):
            self.installed = "real"
            return "hook"

    obj = Installing()
    assert obj.installed == "hook"
    for _ in range(12000):
        seen = obj.installed
    assert seen == "real"


main()
# These exercise pins distinct from main's two invalidations while keeping
# every loop independently traceable.
type_shadow_case()
receiver_binding_case()
raising_hook_case()
self_installing_hook_case()
