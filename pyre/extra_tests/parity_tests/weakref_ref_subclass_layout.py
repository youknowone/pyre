# CPython-suite gap: weakref subclass tests do not combine callbacks with all storage shapes.
# parity-tests reason: this guards the translated W_Weakref payload and subclass carrier layout.

import gc
from weakref import WeakValueDictionary, ref


class Value:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Value) and self.value == other.value

    def __hash__(self):
        return hash(self.value)


class PlainRef(ref):
    pass


class EmptyRef(ref):
    __slots__ = ()


class KeyedRef(ref):
    __slots__ = ("key",)


PRIVATE_FIELDS = ("w_obj_weak", "w_callable", "w_hash", "w_slots")


def collect():
    for _ in range(3):
        gc.collect()


def check_ref_subclass(ref_type):
    callbacks = []

    def callback(w_ref):
        callbacks.append(w_ref() is None)

    value = Value((ref_type.__name__, 42))
    equal_value = Value((ref_type.__name__, 42))
    w_ref = ref_type(value, callback)
    equal_ref = ref(equal_value)

    assert w_ref() is value
    assert w_ref.__callback__ is callback
    assert hash(w_ref) == hash(value)
    assert w_ref == equal_ref

    if ref_type is PlainRef:
        w_ref.marker = "plain"
        assert w_ref.__dict__ == {"marker": "plain"}
    else:
        assert not hasattr(w_ref, "__dict__")

    if ref_type is KeyedRef:
        w_ref.key = ("module", "name")
        assert w_ref.key == ("module", "name")
        w_ref.key = ("module", "renamed")
        assert w_ref.key == ("module", "renamed")
        assert w_ref() is value
        del w_ref.key
        try:
            w_ref.key
        except AttributeError:
            pass
        else:
            raise AssertionError("deleted slot remained readable")

    for name in PRIVATE_FIELDS:
        assert not hasattr(w_ref, name)
        if hasattr(w_ref, "__dict__"):
            assert name not in w_ref.__dict__

    del value
    collect()
    assert w_ref() is None
    assert callbacks == [True]


for ref_type in (PlainRef, EmptyRef, KeyedRef):
    check_ref_subclass(ref_type)


values = WeakValueDictionary()
key = ("namespace", "spam")
value = Value("live")
values[key] = value
assert values[key] is value
stored_ref = values.data[key]
assert stored_ref.key == key
assert stored_ref() is value
del value
collect()
assert key not in values

print("OK")
