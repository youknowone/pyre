"""`os.stat` keeps `follow_symlinks` keyword-only in Python 3.14."""

import os
import shutil
import tempfile


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

# `follow_symlinks=False` has to reach the link itself, which the checks
# above cannot observe: `.` is not a symlink, so both calls describe the
# same file whether or not the flag is honored.
#
# Creating one is skipped on Windows, where it needs a privilege the test
# cannot assume.
if os.name != "nt":
    root = tempfile.mkdtemp()
    try:
        target = os.path.join(root, "target")
        link = os.path.join(root, "link")
        with open(target, "w") as stream:
            stream.write("0123456789")
        os.symlink(target, link)
        assert os.stat(link).st_ino == os.stat(target).st_ino
        assert os.stat(link, follow_symlinks=False).st_ino != os.stat(target).st_ino
        assert os.stat(link, follow_symlinks=False).st_ino == os.lstat(link).st_ino
        assert os.stat(link).st_size == 10
        assert os.stat(link, follow_symlinks=False).st_size != 10
    finally:
        shutil.rmtree(root)

print("OK")
