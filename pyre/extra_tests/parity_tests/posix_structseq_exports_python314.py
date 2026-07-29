"""CPython 3.14 POSIX structseq type exports and pickle parity."""

import os
import pickle


values = [os.stat(os.curdir)]
if hasattr(os, "statvfs"):
    values.append(os.statvfs(os.curdir))
if hasattr(os, "times"):
    values.append(os.times())
if hasattr(os, "uname"):
    values.append(os.uname())

for value in values:
    cls = type(value)
    assert getattr(os, cls.__name__) is cls
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(value, protocol))
        assert type(restored) is cls
        assert restored == value

print("OK")
