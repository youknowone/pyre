"""CPython 3.14 set-element serialization notes."""

import pickle


class CustomError(Exception):
    pass


class Unpickleable:
    def __reduce__(self):
        raise CustomError


value = {Unpickleable()}
frozen = frozenset({frozenset({Unpickleable()})})

for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    try:
        pickle.dumps(value, protocol=protocol)
    except CustomError as exc:
        if protocol >= 4:
            notes = ["when serializing set element"]
        else:
            notes = [
                "when serializing list item 0",
                "when serializing tuple item 0",
                "when serializing set reconstructor arguments",
            ]
        assert exc.__notes__ == notes, (protocol, exc.__notes__)
    else:
        raise AssertionError(f"unpickleable set element was accepted: {protocol=}")

    try:
        pickle.dumps(frozen, protocol=protocol)
    except CustomError as exc:
        if protocol >= 4:
            notes = [
                "when serializing frozenset element",
                "when serializing frozenset element",
            ]
        else:
            notes = [
                "when serializing list item 0",
                "when serializing tuple item 0",
                "when serializing frozenset reconstructor arguments",
                "when serializing list item 0",
                "when serializing tuple item 0",
                "when serializing frozenset reconstructor arguments",
            ]
        assert exc.__notes__ == notes, (protocol, exc.__notes__)
    else:
        raise AssertionError(f"unpickleable frozenset element was accepted: {protocol=}")

print("OK")
