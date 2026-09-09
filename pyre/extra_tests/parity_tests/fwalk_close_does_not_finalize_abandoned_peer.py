# An abandoned fwalk generator keeps its dirfds until GC. close() of a
# different fwalk must not run that peer's finally -- otherwise
# test_os.FwalkTests.test_fd_finalization sees extra closes
# (AssertionError: 4 != 7) after test_walk_symlink leaves a generator
# live. CPython drops the peer on the rebind; pypy3 leaves it until a
# later collection and still does not finalize it from this close().
import os
import shutil
import tempfile

if not hasattr(os, "fwalk"):
    print("OK")
    raise SystemExit(0)


def getfd():
    fd = os.dup(1)
    os.close(fd)
    return fd


root = tempfile.mkdtemp()
try:
    os.makedirs(os.path.join(root, "a", "b"))
    held = os.fwalk(root, follow_symlinks=True)
    next(held)
    del held

    old = getfd()
    it = os.fwalk(root, topdown=True)
    next(it)
    it.close()
    after = getfd()
    assert after == old, (after, old)
finally:
    shutil.rmtree(root, ignore_errors=True)

print("OK")
