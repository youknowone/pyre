"""CPython 3.14 nested reducer serialization notes."""

import pickle


class CustomError(Exception):
    pass


class Unpickleable:
    def __call__(self, *args, **kwargs):
        pass

    def __reduce__(self):
        raise CustomError


class Reduced:
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return self.value


try:
    pickle.dumps(Reduced((Unpickleable(), ())), protocol=5)
except CustomError as exc:
    assert exc.__notes__ == [
        "when serializing Reduced reconstructor",
        "when serializing Reduced object",
    ]
else:
    raise AssertionError("unpickleable reconstructor was accepted")

try:
    pickle.dumps(Reduced((print, (1, 2, Unpickleable()))), protocol=5)
except CustomError as exc:
    assert exc.__notes__ == [
        "when serializing tuple item 2",
        "when serializing Reduced reconstructor arguments",
        "when serializing Reduced object",
    ]
else:
    raise AssertionError("unpickleable reconstructor argument was accepted")

print("OK")
