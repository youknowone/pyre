"""CPython 3.14 method-descriptor pickle parity."""

import pickle


def function():
    pass


for descriptor in (staticmethod(function), classmethod(function)):
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        try:
            pickle.dumps(descriptor, protocol)
        except TypeError as exc:
            assert f"cannot pickle '{type(descriptor).__name__}' object" in str(exc)
        else:
            raise AssertionError((descriptor, protocol))

print("OK")
