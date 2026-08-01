"""`os.stat` keeps `follow_symlinks` keyword-only in Python 3.14."""

import os


assert os.stat(".", follow_symlinks=False).st_ino == os.stat(".").st_ino
try:
    os.stat(".", False)
except TypeError:
    pass
else:
    raise AssertionError("follow_symlinks accepted positionally")
try:
    os.stat(".", unknown=True)
except TypeError:
    pass
else:
    raise AssertionError("unknown os.stat keyword was accepted")

print("OK")
