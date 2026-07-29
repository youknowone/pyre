"""CPython 3.14 protocol-0 STRING trailing escape validation."""

import pickle


try:
    pickle.loads(b"S'\\'\n.")
except ValueError as exc:
    assert str(exc) == "Trailing \\ in string"
else:
    raise AssertionError("accepted a STRING ending in a backslash")

print("OK")
