"""OS boundaries take the filesystem encoding, not a strict UTF-8 encode.

`os.system` (interp_posix.py:815 `command='fsencode'`), the `posix_spawn`
argv/envp entries (interp_posix.py:1742 `space.fsencode_w(w_arg)`) and an
AF_UNIX address (interp_socket.py:157-159 `space.fsencode(w_address)`) all
carry OS bytes, so a name spelling a byte with no UTF-8 form must survive
instead of raising UnicodeEncodeError.
"""

import os
import sys

# Every boundary below is POSIX-only, and `_socket` is not built on Windows,
# so the platform check has to run before `socket` is imported.
if sys.platform == "win32":
    print("OK")
    raise SystemExit

import socket
import subprocess
import tempfile


# `os.system`: the surrogate escape reaches the shell as byte 0xff.  The
# command is a no-op either way; what is under test is that encoding the
# argument does not raise.
os.system("true 'x\udcff'")

# `posix_spawn`: a surrogate-escaped argv entry, which `true` ignores.
TRUE = "/usr/bin/true" if os.path.exists("/usr/bin/true") else "/bin/true"

pid = os.posix_spawn(TRUE, [TRUE, "x\udcff"], {})
os.waitpid(pid, 0)

# and a surrogate-escaped environment value.
pid = os.posix_spawn(TRUE, [TRUE], {"PYRE_X": "v\udcff"})
os.waitpid(pid, 0)

for call in (
    lambda: os.system("true a\x00b"),
    lambda: os.posix_spawn(TRUE, [TRUE, "a\x00b"], {}),
    lambda: open("a\x00b"),
):
    try:
        call()
    except ValueError as exc:
        assert "embedded null byte" in str(exc), str(exc)
    else:
        raise AssertionError("embedded null byte was accepted")

# `interp_socket.py:157-159` deliberately uses raw `space.fsencode`: a leading
# null byte names Linux's abstract namespace and must reach the OS unchanged.
name = "\x00pyre-os-boundary-%d" % os.getpid()
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    try:
        s.bind(name)
    except ValueError:
        raise AssertionError("AF_UNIX abstract path rejected its null") from None
    except OSError:
        pass
    else:
        assert s.getsockname() == name.encode(), s.getsockname()
finally:
    s.close()

# AF_UNIX: bind to a path whose last byte has no UTF-8 spelling.  Whether the
# kernel accepts it is a platform question — macOS refuses a non-UTF-8 path
# with EILSEQ — so the assertion is that the *encoding* is not what refuses:
# the address must reach the OS.  Where the bind does succeed, the name read
# back proves the bytes were not mangled on the way.
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "s\udcff")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            s.bind(path)
        except UnicodeEncodeError:
            raise AssertionError("AF_UNIX path was encoded strictly") from None
        except OSError:
            pass
        else:
            assert s.getsockname() == path, s.getsockname()
    finally:
        s.close()

# `socket.if_nametoindex` reads an interface name the same way.  A surrogate
# escape must reach the OS, which answers ENODEV/ENXIO for a name no interface
# carries; only the encoding is under test.
try:
    socket.if_nametoindex("lo\udcff")
except UnicodeEncodeError:
    raise AssertionError("interface name was encoded strictly") from None
except OSError:
    pass

# The `'fsencode'` converter accepts bytes and `__fspath__` as well as str
# (gateway.py:365 `space.fsencode_w`), so these boundaries do too.
assert os.system(b"true") == 0
os.waitpid(os.posix_spawn(TRUE, [TRUE.encode()], {}), 0)
index, name = socket.if_nameindex()[0]
assert socket.if_nametoindex(name.encode()) == index

# The encoded path has to reach the syscall as bytes.  Were it folded into text
# first, every byte with no UTF-8 spelling would become U+FFFD and distinct
# names would alias onto one another: a lookup of `b"\xffvictim"` would answer
# for the file actually called `"�victim"`.  Plant that file and check the
# lookup misses it.
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "�victim"), "wb") as fp:
        fp.write(b"victim")
    probe = os.path.join(os.fsencode(d), b"\xffvictim")
    assert not os.path.exists(probe), "a non-UTF-8 path aliased onto U+FFFD"
    for call in (lambda: os.stat(probe), lambda: open(probe)):
        try:
            call()
        except OSError:
            pass
        else:
            raise AssertionError("a non-UTF-8 path aliased onto U+FFFD")

# `wrap_oserror2(space, e, w_path)` reports the path object the call was given,
# so a bytes argument reports bytes and a str keeps its surrogate escapes rather
# than both being re-spelled through a lossy decode.
with tempfile.TemporaryDirectory() as d:
    for arg in (
        os.path.join(d, "absent\udcff"),
        os.path.join(os.fsencode(d), b"absent\xff"),
    ):
        for call in (os.stat, open):
            try:
                call(arg)
            except OSError as exc:
                assert exc.filename == arg, (ascii(exc.filename), ascii(arg))
            else:
                raise AssertionError("an absent path was found")


class _Spelled:
    """An `os.PathLike` that is not itself a path."""

    def __init__(self, name):
        self.name = name

    def __fspath__(self):
        return self.name


# `interp_posix.py:211-219 _unwrap_path` calls `__fspath__` and keeps its
# result, so the name a failure reports is the path the object spelled, never
# the object that spelled it.
with tempfile.TemporaryDirectory() as d:
    for spelled in (
        os.path.join(d, "absent\udcff"),
        os.path.join(os.fsencode(d), b"absent\xff"),
    ):
        for call in (os.stat, os.listdir, open):
            try:
                call(_Spelled(spelled))
            except OSError as exc:
                assert exc.filename == spelled, (ascii(exc.filename), ascii(spelled))
            else:
                raise AssertionError("an absent path was found")

# `interp_posix.py:1814-1817` wraps a failed exec with no filename at all.
try:
    os.execv("/nonexistent-pyre-boundary", ["x"])
except OSError as exc:
    assert exc.filename is None, ascii(exc.filename)
else:
    raise AssertionError("execv of an absent program returned")

# `interp_posix.py:1762-1769` rejects `=` only after index zero, so the `=C:`
# form Windows uses for a working directory is a legal key.
os.waitpid(os.posix_spawn(TRUE, [TRUE], {"=C:": "v"}), 0)

# `os.putenv` takes its two halves through the same converter, and
# `posix.environ` hands entries back as bytes, so a value spelling a byte with
# no UTF-8 form has to survive the write rather than folding to U+FFFD.
NAME = "PYRE_OS_BOUNDARY_%d" % os.getpid()
os.putenv(NAME, "v\udcff")
try:
    read_back = subprocess.run(
        [sys.executable, "-c", "import os,sys; sys.stdout.buffer.write(os.environb[b'%s'])" % NAME],
        capture_output=True,
    )
    if read_back.returncode == 0:
        assert read_back.stdout == b"v\xff", read_back.stdout
finally:
    os.unsetenv(NAME)

print("OK")
