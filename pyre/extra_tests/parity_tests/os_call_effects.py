"""Every `os` call named here changes something, and the change is visible.

A name the module answers with `None` and no effect passes `hasattr` and
returns without raising, so the code that calls it — `shutil`, `tempfile`,
`subprocess` — carries on as though the file had been truncated, the working
directory changed, the descriptor made inheritable.  Each call below is
followed by the state it is supposed to have left behind.

The permission bits Windows keeps are the read-only attribute alone, which
`0o444` and `0o666` are the two spellings of, so the modes asserted here are
the ones both platforms report.
"""

import os
import sys
import tempfile

WIN32 = sys.platform == "win32"

base = tempfile.mkdtemp(prefix="pyre_effects_")
os.chdir(base)
# The name a temporary directory is made under can be reached by a shorter
# path than the one it is spelled with, so the directory that `getcwd`
# answers is the one the filesystem resolves.
base = os.getcwd()
a_dir = os.path.join(base, "d")
os.mkdir(a_dir)
a_file = os.path.join(base, "f")
with open(a_file, "w") as fp:
    fp.write("hello")

# os.chdir
os.chdir(a_dir)
assert os.getcwd() == os.path.realpath(a_dir), os.getcwd()
os.chdir(base)
assert os.getcwd() == base, os.getcwd()

# os.chmod / os.fchmod, read back through the mode `os.stat` reports.
os.chmod(a_file, 0o444)
assert os.stat(a_file).st_mode & 0o777 == 0o444, oct(os.stat(a_file).st_mode)
os.chmod(a_file, 0o666)
assert os.stat(a_file).st_mode & 0o777 == 0o666, oct(os.stat(a_file).st_mode)

# Windows changes a file's attributes through its handle, and a handle opened
# for reading alone is not one such a change can be asked of.
fd = os.open(a_file, os.O_RDWR)
os.fchmod(fd, 0o444)
assert os.stat(a_file).st_mode & 0o777 == 0o444, oct(os.stat(a_file).st_mode)
os.fchmod(fd, 0o666)
os.close(fd)

# os.access.  A file the owner cannot write is the one case Windows can
# answer `False` to for a name that exists — and the superuser a POSIX host
# may be running as is exempt from the bits entirely.
assert os.access(a_file, os.F_OK)
assert not os.access(os.path.join(base, "missing"), os.F_OK)
if WIN32 or (hasattr(os, "geteuid") and os.geteuid() != 0):
    os.chmod(a_file, 0o444)
    assert not os.access(a_file, os.W_OK)
    os.chmod(a_file, 0o666)
    assert os.access(a_file, os.W_OK)

# os.truncate / os.ftruncate
os.truncate(a_file, 3)
assert os.stat(a_file).st_size == 3, os.stat(a_file).st_size
fd = os.open(a_file, os.O_RDWR)
os.ftruncate(fd, 1)
os.close(fd)
assert os.stat(a_file).st_size == 1, os.stat(a_file).st_size

# os.link — a second name for the one file, so what one name reads the other
# wrote.  Not every filesystem holds more than one name per file; the ones
# that do not say so rather than quietly keeping one.
link = os.path.join(base, "l")
try:
    os.link(a_file, link)
except (OSError, NotImplementedError):
    pass
else:
    assert os.path.exists(link)
    with open(a_file, "wb") as fp:
        fp.write(b"linked")
    with open(link, "rb") as fp:
        assert fp.read() == b"linked"
    os.remove(link)

# os.pipe, whose two descriptors carry bytes one way and are closed to
# children — which `os.set_inheritable` is what changes.
read_fd, write_fd = os.pipe()
assert os.write(write_fd, b"xy") == 2
assert os.read(read_fd, 2) == b"xy"
assert os.get_inheritable(read_fd) is False
os.set_inheritable(read_fd, True)
assert os.get_inheritable(read_fd) is True
os.set_inheritable(read_fd, False)
assert os.get_inheritable(read_fd) is False
os.close(read_fd)
os.close(write_fd)

# os.umask answers with the mask it replaces, so the mask it was given is the
# one the next call gives back — of the bits the platform keeps, which on
# Windows are the two it has, `S_IREAD | S_IWRITE`.
previous = os.umask(0o600)
assert os.umask(previous) == 0o600

# os.dup names the same open file, and the copy is closed to children
# whether or not the original was.
with open(a_file, "wb") as fp:
    fp.write(b"duped")
fd = os.open(a_file, os.O_RDONLY)
copy = os.dup(fd)
assert copy != fd
assert os.get_inheritable(copy) is False
assert os.read(copy, 5) == b"duped"

# os.dup2 names a descriptor of its caller's choosing, and hands it to
# children unless it is told not to.
os.dup2(fd, copy)
assert os.get_inheritable(copy) is True
os.dup2(fd, copy, False)
assert os.get_inheritable(copy) is False
os.close(copy)
os.close(fd)

# The process-wide answers, which are not `None` on any host that runs this.
assert isinstance(os.getppid(), int) and os.getppid() > 0
assert isinstance(os.cpu_count(), int) and os.cpu_count() > 0
try:
    # A process with no session behind it has no login name to give.
    login = os.getlogin()
except OSError:
    pass
else:
    assert isinstance(login, str) and login, login

os.chdir(tempfile.gettempdir())
os.remove(a_file)
os.rmdir(a_dir)
os.rmdir(base)
print("OK")
