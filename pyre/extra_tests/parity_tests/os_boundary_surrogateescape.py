"""OS boundaries take the filesystem encoding, not a strict UTF-8 encode.

`os.system` (interp_posix.py:815 `command='fsencode'`), the `posix_spawn`
argv/envp entries (interp_posix.py:1742 `space.fsencode_w(w_arg)`) and an
AF_UNIX address (interp_socket.py:157-159 `space.fsencode(w_address)`) all
carry OS bytes, so a name spelling a byte with no UTF-8 form must survive
instead of raising UnicodeEncodeError.
"""

import os
import socket
import sys
import tempfile

if sys.platform == "win32":
    print("OK")
    raise SystemExit


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
assert socket.if_nametoindex(socket.if_indextoname(1).encode()) == 1

print("OK")
