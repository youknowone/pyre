# pyre-check: gate=1
# A slot may shadow an inherited method with an unset member, so that the
# class's `__getattr__` installs one on first read.  The member's
# AttributeError has to reach the hook whether or not instances carry a
# `__dict__`, which is what `_handle_getattribute` wraps.


class Hook:
    __slots__ = ()

    def __getattr__(self, key):
        return "hook(%s)" % key


class DictLess(Hook, str):
    __slots__ = ("lower",)


class DictCarrying(str):
    __slots__ = ("lower",)

    def __getattr__(self, key):
        return "hook(%s)" % key


for cls in (DictLess, DictCarrying):
    receiver = cls("AbC")
    assert type(receiver).__dict__["lower"].__class__.__name__ == "member_descriptor"
    assert receiver.lower == "hook(lower)", (cls, receiver.lower)
    # A name the class does not shadow reaches the hook too.
    assert receiver.absent == "hook(absent)"
    # The inherited method answers again once the slot holds a value, and the
    # hook is back after the slot is emptied.
    receiver.lower = "stored"
    assert receiver.lower == "stored"
    del receiver.lower
    assert receiver.lower == "hook(lower)"
    # Reading the slot leaves the string itself untouched.
    assert str.lower(receiver) == "abc"


# Without a hook the member's own AttributeError is what is reported.
class NoHook(str):
    __slots__ = ("lower",)


try:
    NoHook("AbC").lower
except AttributeError as exc:
    assert "lower" in str(exc), exc
else:
    raise AssertionError("an unset slot answered")
