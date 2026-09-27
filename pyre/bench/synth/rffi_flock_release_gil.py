# pyre-check: skip-backends=cranelift
# cranelift runs the core build, which has no `pyre-module` and so no `fcntl`.
# pyre-check: skip-cpython
# Hot flock so the trace of `c_flock` records call_release_gil.
import errno
import fcntl
import os
import tempfile

fd, path = tempfile.mkstemp()
try:
    n = 0
    acc = 0
    err = 0
    while n < 20001:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
        except OSError as e:
            err = e.errno
            break
        fcntl.flock(fd, fcntl.LOCK_UN)
        acc = acc + n
        n = n + 1
        if n == 20000:
            os.close(fd)
    if err != errno.EBADF:
        raise SystemExit("errno %s" % err)
    print(acc)
    print(err)
finally:
    try:
        os.close(fd)
    except OSError:
        pass
    os.unlink(path)
