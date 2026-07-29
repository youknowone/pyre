"""CPython 3.14 NEWOBJ reduce-value validation parity."""

import copyreg
import pickle


class Reduced:
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return self.value


cases = [
    (
        (copyreg.__newobj__, ()),
        "__newobj__ expected at least 1 argument, got 0",
    ),
    (
        (copyreg.__newobj__, [Reduced]),
        "second item of the tuple returned by __reduce__ must be a tuple, not list",
    ),
    (
        (copyreg.__newobj_ex__, ()),
        "__newobj_ex__ expected 3 arguments, got 0",
    ),
    (
        (copyreg.__newobj_ex__, (Reduced, 42, {})),
        "second argument to __newobj_ex__() must be a tuple, not int",
    ),
    (
        (copyreg.__newobj_ex__, (Reduced, (), [])),
        "third argument to __newobj_ex__() must be a dict, not list",
    ),
]

for value, message in cases:
    try:
        pickle.dumps(Reduced(value), protocol=5)
    except pickle.PicklingError as exc:
        assert str(exc) == message
        assert exc.__notes__ == ["when serializing Reduced object"]
    else:
        raise AssertionError(f"invalid reducer accepted: {value!r}")

print("OK")
