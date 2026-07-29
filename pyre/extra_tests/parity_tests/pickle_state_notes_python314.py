"""CPython 3.14 reducer state serialization notes."""

import pickle


class CustomError(Exception):
    pass


class Unpickleable:
    def __reduce__(self):
        raise CustomError


class UnpickleableCallable(Unpickleable):
    def __call__(self, *args, **kwargs):
        pass


class State:
    def __init__(self, state=None):
        self.state = state

    def __reduce__(self):
        return type(self), (), self.state


class Reduced:
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return self.value


cases = [
    (
        State(Unpickleable()),
        ["when serializing State state", "when serializing State object"],
    ),
    (
        Reduced((print, (), "state", None, None, UnpickleableCallable())),
        ["when serializing Reduced state setter", "when serializing Reduced object"],
    ),
    (
        Reduced((print, (), Unpickleable(), None, None, print)),
        ["when serializing Reduced state", "when serializing Reduced object"],
    ),
]

for value, notes in cases:
    try:
        pickle.dumps(value, protocol=5)
    except CustomError as exc:
        assert exc.__notes__ == notes
    else:
        raise AssertionError(f"unpickleable reducer state was accepted: {value!r}")

print("OK")
