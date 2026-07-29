"""CPython 3.14 list-item serialization notes."""

import pickle


class CustomError(Exception):
    pass


class Unpickleable:
    def __reduce__(self):
        raise CustomError


class Reduced:
    def __reduce__(self):
        return list, (), None, iter([1, 2, Unpickleable()])


cases = [
    (
        [1, [2, 3, Unpickleable()]],
        ["when serializing list item 2", "when serializing list item 1"],
    ),
    (
        Reduced(),
        ["when serializing Reduced item 2", "when serializing Reduced object"],
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
                f"unpickleable list item was accepted: {protocol=}, {value=!r}"
            )

print("OK")
