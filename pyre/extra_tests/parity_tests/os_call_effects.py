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
import shutil
import stat
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

# os.truncate / os.ftruncate.  `truncate` takes either a name or an open
# descriptor — its path argument is `path_t(allow_fd=…)`, which is also what
# puts it in `supports_fd`.
os.truncate(a_file, 3)
assert os.stat(a_file).st_size == 3, os.stat(a_file).st_size
fd = os.open(a_file, os.O_RDWR)
os.ftruncate(fd, 1)
assert os.stat(a_file).st_size == 1, os.stat(a_file).st_size
os.truncate(fd, 2)
assert os.fstat(fd).st_size == 2, os.fstat(fd).st_size
os.close(fd)
assert os.truncate in os.supports_fd

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

# A directory descriptor to resolve either name against is honoured where the
# platform has `linkat` and refused where it does not — never ignored, which
# would link a same-named file from the process's own directory instead.
if os.link in os.supports_dir_fd:
    with open(os.path.join(a_dir, "inner"), "wb") as fp:
        fp.write(b"inner")
    dir_fd = os.open(a_dir, os.O_RDONLY)
    try:
        os.link("inner", "l2", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    # The working directory is `base`, so a link that landed there is one whose
    # names were resolved against it rather than against the descriptor.
    assert os.path.exists(os.path.join(a_dir, "l2"))
    assert not os.path.exists(os.path.join(base, "l2"))
    os.remove(os.path.join(a_dir, "l2"))
    os.remove(os.path.join(a_dir, "inner"))
else:
    try:
        os.link(a_file, os.path.join(base, "l3"), src_dir_fd=0)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("link ignored src_dir_fd")
    assert not os.path.exists(os.path.join(base, "l3"))

# Whether the source symlink is followed or linked is the platform's to say;
# what it cannot do is take the argument and ignore it.
l4 = os.path.join(base, "l4")
if os.link in os.supports_follow_symlinks:
    os.link(a_file, l4, follow_symlinks=True)
    os.remove(l4)
else:
    try:
        os.link(a_file, l4, follow_symlinks=True)
    except (NotImplementedError, OSError):
        pass
    else:
        raise AssertionError("link ignored follow_symlinks")
    assert not os.path.exists(l4)

# Both descriptors and `follow_symlinks` are keyword-only, so a third
# positional argument is not one of them being passed.
try:
    os.link(a_file, os.path.join(base, "l5"), 0)
except TypeError:
    pass
else:
    raise AssertionError("link took a third positional argument")
assert not os.path.exists(os.path.join(base, "l5"))

# os.symlink — a name that resolves to another, which `lstat` reports as the
# link and `stat` as what it points at.  The target is stored as it was given,
# so a relative one reads back the way it was written.  Creating a link at all
# is a privilege on Windows that the account may not hold, so a refusal is an
# answer too.
sym = os.path.join(base, "s")
try:
    os.symlink("f", sym)
except (OSError, NotImplementedError):
    pass
else:
    assert os.path.islink(sym), sym
    assert os.readlink(sym) == "f", os.readlink(sym)
    assert stat.S_ISLNK(os.lstat(sym).st_mode)
    assert stat.S_ISREG(os.stat(sym).st_mode)
    with open(sym, "rb") as fp, open(a_file, "rb") as original:
        assert fp.read() == original.read()
    # Removing the link removes the link.
    os.remove(sym)
    assert os.path.exists(a_file)
    # A link to a directory is a directory to whatever follows it, and the
    # execute bits belong to the link's own format.
    dsym = os.path.join(base, "ds")
    os.symlink("d", dsym, target_is_directory=True)
    assert stat.S_ISLNK(os.lstat(dsym).st_mode)
    assert stat.S_ISDIR(os.stat(dsym).st_mode)
    assert os.lstat(dsym).st_mode & 0o111 == 0o111, oct(os.lstat(dsym).st_mode)
    # `unlink` takes the link whichever it names, though what Windows has to
    # call to take a directory one is `RemoveDirectory`.
    os.remove(dsym)
    assert os.path.isdir(a_dir)

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

# os.system runs the command and answers with the status the interpreter
# exited on — a wait status where the platform has those, the exit code itself
# on Windows.
marker = os.path.join(base, "ran")
if WIN32:
    status = os.system('echo ran> "%s" & exit 3' % marker)
    code = status
else:
    status = os.system('echo ran > "%s"; exit 3' % marker)
    code = os.waitstatus_to_exitcode(status)
assert code == 3, (status, code)
assert os.path.exists(marker), marker
os.remove(marker)

# os.times counts the process's own time, which is a float and not negative.
times = os.times()
assert len(times) == 5, times
assert times.user >= 0.0 and times.system >= 0.0, times
assert times[:2] == (times.user, times.system), times

# os.waitpid has nothing to wait for, which is `ECHILD` and not silence.
try:
    os.waitpid(-424242, 0)
except ChildProcessError as exc:
    assert exc.errno == 10, exc.errno
else:
    raise AssertionError("waitpid on no child must fail")

# os.get_terminal_size measures the terminal a descriptor names; one that
# names no descriptor at all has none to measure.  (Whether *this* process has
# a terminal is the caller's business — `shutil.get_terminal_size` is where the
# fallback lives, and it is reached by catching this.)
try:
    os.get_terminal_size(999)
except OSError as exc:
    assert exc.errno == 9, exc.errno
else:
    raise AssertionError("get_terminal_size on a bad descriptor must fail")
assert shutil.get_terminal_size().columns > 0

if WIN32:
    # os.listdrives names the roots, `C:\` among them.
    drives = os.listdrives()
    assert drives and all(d.endswith(":\\") for d in drives), drives
    assert os.path.splitdrive(sys.executable)[0] + "\\" in drives, drives
    # Every volume is mounted under zero or more of those names.
    for volume in os.listvolumes():
        assert volume.startswith("\\\\?\\Volume"), volume
        for mount in os.listmounts(volume):
            assert os.path.isabs(mount), mount

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
