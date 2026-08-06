"""`os` is a thin wrapper over the module the platform implements it with, and
which names that module carries is how the standard library decides what the
platform can do.

`os.py` does `from posix import *` (or `from nt import *`) and then probes what
arrived: `_exists("fork")` picks the POSIX spawn implementation,
`shutil.disk_usage` picks its own on `hasattr(os, 'statvfs')`, and
`multiprocessing` picks a start method on `hasattr(os, 'fork')`.  A name bound
to a stub that answers `None` is therefore not a harmless extra — it wins the
branch on a platform that cannot serve it, and the call it was chosen for
either does nothing or fails much later.

So the assertions here are about presence, not behaviour: `nt` must not answer
to the POSIX-only calls, and the constants it does carry must carry the C
runtime's values rather than a placeholder.
"""

import errno
import os
import sys

WIN32 = sys.platform == "win32"

# The module `os` is implemented by, which is the one whose surface decides.
impl = __import__(os.name)
assert impl.__name__ == ("nt" if WIN32 else "posix"), impl.__name__

# Calls that exist only where there are processes in the POSIX sense: ids,
# process groups, terminals, wait-status decoding, the scheduler.  The list is
# deliberately the portable core -- `pipe2`, `dup3` and `fdatasync` are Linux's
# and not every POSIX host has them, so they are not asserted either way.
POSIX_ONLY = [
    "fork",
    "forkpty",
    "wait",
    "getuid",
    "geteuid",
    "getgid",
    "getegid",
    "setuid",
    "setgid",
    "setsid",
    "setpgid",
    "getpgid",
    "getpgrp",
    "setpgrp",
    "getgroups",
    "setgroups",
    "nice",
    "ttyname",
    "ctermid",
    "openpty",
    "tcgetpgrp",
    "tcsetpgrp",
    "killpg",
    "getpriority",
    "setpriority",
    "statvfs",
    "fstatvfs",
    "fchdir",
    "fchown",
    "mkfifo",
    "major",
    "minor",
    "makedev",
    "pathconf",
    "fpathconf",
    "confstr",
    "sysconf",
    "register_at_fork",
    "WIFEXITED",
    "WEXITSTATUS",
    "WIFSIGNALED",
    "WTERMSIG",
    "WIFSTOPPED",
    "WSTOPSIG",
    "WNOHANG",
    "EX_USAGE",
    "EX_IOERR",
    "SCHED_OTHER",
    "PRIO_PROCESS",
    "ST_RDONLY",
    "O_NONBLOCK",
]

if WIN32:
    for name in POSIX_ONLY:
        assert not hasattr(impl, name), name
    # And so the probes that read them answer the way the platform can serve.
    assert not hasattr(os, "fork"), "os.fork on Windows picks the POSIX branch"
    assert not hasattr(os, "statvfs")
    assert not hasattr(os, "getuid")
else:
    # The other direction is the platform's to vary in detail, so only the core
    # every POSIX host has is asserted -- enough to catch the list being
    # gated off on the platform it belongs to.
    for name in ("fork", "wait", "getuid", "geteuid", "setsid", "WIFEXITED"):
        assert hasattr(impl, name), name
    assert hasattr(os, "fork")

# `SEEK_SET` and friends are os.py's own on every platform (`SEEK_SET = 0`);
# neither implementation module carries them.
for name in ("SEEK_SET", "SEEK_CUR", "SEEK_END"):
    assert not hasattr(impl, name), name
assert (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END) == (0, 1, 2)

if WIN32:
    # The spawn modes are the C runtime's `_P_*` (process.h).  A set of these
    # bound to a single placeholder value makes `P_NOWAIT` mean `P_WAIT`.
    assert (impl.P_WAIT, impl.P_NOWAIT, impl.P_OVERLAY, impl.P_NOWAITO, impl.P_DETACH) == (
        0,
        1,
        2,
        3,
        4,
    )
    assert isinstance(impl.TMP_MAX, int) and impl.TMP_MAX > 0, impl.TMP_MAX
    # The LoadLibraryEx search flags `os.add_dll_directory` and `ctypes` pass on.
    assert impl._LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR == 0x100
    assert impl._LOAD_LIBRARY_SEARCH_APPLICATION_DIR == 0x200
    assert impl._LOAD_LIBRARY_SEARCH_USER_DIRS == 0x400
    assert impl._LOAD_LIBRARY_SEARCH_SYSTEM32 == 0x800
    assert impl._LOAD_LIBRARY_SEARCH_DEFAULT_DIRS == 0x1000

# os.strerror reads the same table OSError does, so the two agree on a code.
for code in (errno.ENOENT, errno.EBADF, errno.EACCES):
    message = os.strerror(code)
    assert isinstance(message, str) and message, (code, message)
    assert OSError(code, message).strerror == message
assert os.strerror(errno.ENOENT) == "No such file or directory"
try:
    os.strerror("2")
except TypeError:
    pass
else:
    raise AssertionError("strerror must reject a non-integer code")

# os.closerange closes the half-open range and reports nothing, including for
# the descriptors in it that were never open.
read_fd, write_fd = os.pipe()
low, high = min(read_fd, write_fd), max(read_fd, write_fd)
assert os.closerange(low, high + 1) is None
for fd in (read_fd, write_fd):
    try:
        os.fstat(fd)
    except OSError as exc:
        assert exc.errno == errno.EBADF, exc.errno
    else:
        raise AssertionError("closerange left fd %d open" % fd)
# A range naming nothing is not an error -- it is the ordinary case.
assert os.closerange(high + 400, high + 405) is None

print("OK")
