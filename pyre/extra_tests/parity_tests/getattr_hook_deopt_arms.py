# CPython-suite gap: attribute tests do not hot-loop every receiver/type invalidation arm.
# parity-tests reason: pins __getattr__ deopts without adding guard noise to the synth gate.

"""A compiled __getattr__ fold observes receiver, type, and hook mutations."""


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


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


type_shadow_case()
receiver_binding_case()
raising_hook_case()
self_installing_hook_case()
print("OK")
