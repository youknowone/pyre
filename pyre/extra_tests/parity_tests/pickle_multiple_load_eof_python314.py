"""CPython 3.14 repeated Unpickler.load EOF parity."""

import io
import pickle


stream = io.BytesIO(pickle.dumps([1, 2]) + pickle.dumps({"three": 3}))
unpickler = pickle.Unpickler(stream)
assert unpickler.load() == [1, 2]
assert unpickler.load() == {"three": 3}
try:
    unpickler.load()
except EOFError as exc:
    assert exc.args == ("Ran out of input",)
else:
    raise AssertionError("Unpickler.load accepted an exhausted stream")

print("OK")
