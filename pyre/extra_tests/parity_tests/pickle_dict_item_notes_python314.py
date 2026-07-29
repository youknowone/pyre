"""CPython 3.14 dict-item serialization notes."""

import pickle


class CustomError(Exception):
    pass


class Unpickleable:
    def __reduce__(self):
        raise CustomError


class Reduced:
    def __reduce__(self):
        return dict, (), None, None, iter([("a", Unpickleable())])


cases = [
    (
        {"a": {"b": Unpickleable()}},
        ["when serializing dict item 'b'", "when serializing dict item 'a'"],
    ),
    (
        Reduced(),
        ["when serializing Reduced item 'a'", "when serializing Reduced object"],
    ),
]

for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    for value, notes in cases:
        try:
            pickle.dumps(value, protocol=protocol)
        except CustomError as exc:
            assert exc.__notes__ == notes, (protocol, exc.__notes__)
        else:
            raise AssertionError(
                f"unpickleable dict item was accepted: {protocol=}, {value=!r}"
            )

print("OK")
