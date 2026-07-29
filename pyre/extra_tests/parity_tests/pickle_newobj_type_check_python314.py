"""CPython 3.14 NEWOBJ class-argument validation parity."""

import pickle


class NewObj:
    pass


original = NewObj
payload = pickle.dumps(NewObj(), protocol=4)
NewObj = 42
try:
    pickle.loads(payload)
except pickle.UnpicklingError as exc:
    assert str(exc) == "NEWOBJ class argument must be a type, not int"
else:
    raise AssertionError("NEWOBJ accepted a non-type class argument")
finally:
    NewObj = original

print("OK")
