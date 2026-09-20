# pyre-check: gate=1
# pyre-check: platforms=win32
# CPython-suite gap: test_socket's closed-socket cases assert only that an
# OSError is raised, never which code it carries, so the vendored suite passes
# either way.
# parity-tests reason: `RSocket.close` leaves the descriptor at
# INVALID_SOCKET and pyre's stand-in for the call that would follow reported
# the kernel's EBADF on every host.  WinSock answers WSAENOTSOCK instead, and
# `set_error` carries a WinSock code in `.winerror`, which is the attribute
# Windows callers branch on.

"""A closed socket reports the error WinSock would have reported."""

import socket

WSAENOTSOCK = 10038

s = socket.socket()
s.close()

calls = (
    ("send", lambda: s.send(b"x")),
    ("recv", lambda: s.recv(1)),
    ("sendall", lambda: s.sendall(b"x")),
    ("getsockname", lambda: s.getsockname()),
    ("getpeername", lambda: s.getpeername()),
    ("listen", lambda: s.listen(1)),
    ("setsockopt", lambda: s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)),
    ("getsockopt", lambda: s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)),
    ("shutdown", lambda: s.shutdown(socket.SHUT_RDWR)),
)
for name, call in calls:
    try:
        call()
    except OSError as e:
        # The code belongs in `.winerror`; `.errno` is derived from it, and a
        # WinSock code outside the six that have a POSIX twin keeps its own
        # value there.
        assert e.winerror == WSAENOTSOCK, (name, e.winerror, e.errno)
        assert e.errno == WSAENOTSOCK, (name, e.errno)
        assert str(e).startswith("[WinError %d]" % WSAENOTSOCK), (name, str(e))
        # Not one of the errno-specific subclasses: 10038 names no errno.
        assert type(e) is OSError, (name, type(e))
    else:
        raise SystemExit("%s on a closed socket did not fail" % name)

# A socket whose descriptor was detached rather than closed answers the same
# way, because it is the same INVALID_SOCKET.
d = socket.socket()
d.detach()
try:
    d.recv(1)
except OSError as e:
    assert e.winerror == WSAENOTSOCK, (e.winerror, e.errno)
else:
    raise SystemExit("recv on a detached socket did not fail")

print("OK")
