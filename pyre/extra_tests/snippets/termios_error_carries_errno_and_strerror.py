# pyre-check: gate=1
# pyre-check: platforms=darwin,linux
# CPython-suite gap: test_termios does not import, so nothing in the suite
# inspects what a failed termios call raises.
# parity-tests reason: every pyre call site built the message with
# `format!("<syscall>: {e}")` over a `std::io::Error`, which names the syscall
# and appends Rust's `(os error N)`, while `wrap_oserror` and
# `PyErr_SetFromErrno` both spell it with the platform's `strerror` alone.

# pyre-check: pypy-diverges: `termios.error` is not an OSError subclass, so
# `wrap_oserror` hands it all five positional arguments and pypy3 reports
# `(19, 'Operation not supported by device', None, None, None)`;
# `PyErr_SetFromErrno` builds the two-item form.

# `pexpect` and `ptyprocess` match on `e.args[1]` when deciding whether a
# terminal is simply not a terminal, and a message with two extra parts around
# the text never matches.

import os
import termios

fd = os.open(os.devnull, os.O_RDONLY)
try:
    calls = (
        ("tcgetattr", lambda: termios.tcgetattr(fd)),
        ("tcdrain", lambda: termios.tcdrain(fd)),
        ("tcflush", lambda: termios.tcflush(fd, termios.TCIFLUSH)),
        ("tcflow", lambda: termios.tcflow(fd, termios.TCOOFF)),
        ("tcsendbreak", lambda: termios.tcsendbreak(fd, 0)),
    )
    for name, call in calls:
        try:
            call()
        except termios.error as e:
            assert len(e.args) == 2, (name, e.args)
            errno, strerror = e.args
            assert isinstance(errno, int), (name, e.args)
            # The message is the platform's own, with nothing added at either
            # end — no syscall name in front, no error code behind.
            assert strerror == os.strerror(errno), (name, e.args)
            # `termios.error` is registered under a dotted name; `repr` spells
            # the class the way `__name__` does.
            assert repr(e) == "error(%d, %r)" % (errno, strerror), repr(e)
        else:
            raise SystemExit("%s on %s did not fail" % (name, os.devnull))
finally:
    os.close(fd)

# A closed descriptor is the kernel's EBADF, reported the same way.
fd = os.open(os.devnull, os.O_RDONLY)
os.close(fd)
try:
    termios.tcgetattr(fd)
except termios.error as e:
    assert e.args[1] == os.strerror(e.args[0]), e.args
else:
    raise SystemExit("tcgetattr on a closed descriptor did not fail")

print("OK")
