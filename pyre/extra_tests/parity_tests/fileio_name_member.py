# CPython-suite gap: test_fileio.testRepr covers del name, not assignment.
# parity-tests reason: interp_fileio.py interp_member_w w_name drives repr.

"""FileIO.name is the interp_member_w w_name field.

`del f.name` and `f.name = value` update the typed field, so `repr(f)`
follows the live name or switches to the fd form.
"""

import io
import os
import tempfile


path = tempfile.mktemp()
try:
    with open(path, "wb") as fh:
        fh.write(b"x")
    f = io.FileIO(path, "rb")
    print("open-repr", "name=" in repr(f), f.name == path)
    f.name = "renamed"
    print("set-repr", "name='renamed'" in repr(f) or 'name="renamed"' in repr(f), f.name)
    del f.name
    print("del-repr", "fd=" in repr(f), "name=" not in repr(f))
    try:
        f.name
    except AttributeError:
        print("del-get", "AttributeError")
    else:
        print("del-get", "ok")
    f.close()
    print("closed-repr", "[closed]" in repr(f))
finally:
    try:
        os.unlink(path)
    except OSError:
        pass
print("OK")
