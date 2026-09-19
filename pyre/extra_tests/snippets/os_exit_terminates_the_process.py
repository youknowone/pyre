# pyre-check: gate=1
# CPython-suite gap: test_os only reaches `os._exit` through a forked child
# (`test_fork`, POSIX-only), so no vendored module observes the call on a host
# without fork.
# parity-tests reason: os.py finds both names because `install_noop_stubs`
# binds them for every host, and only the POSIX branch replaced them -- on
# Windows `os._exit` returned None and the process ran on, which is the one
# thing the call is documented not to do.

"""`os._exit` ends the process with the status it was given."""

import os
import subprocess
import sys

# A status the interpreter would never produce on its own, so a process that
# ran to the end is distinguishable from one that exited here.
child = subprocess.run(
    [
        sys.executable,
        "-c",
        'import os, sys; sys.stdout.write("before"); sys.stdout.flush();'
        ' os._exit(7); sys.stdout.write("after"); sys.stdout.flush()',
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
assert child.returncode == 7, (child.returncode, child.stdout, child.stderr)
assert child.stdout == b"before", child.stdout

# The status is taken as a C int, so `os._exit` owes a TypeError for an
# argument that is not an integer rather than exiting on whatever it read.
try:
    os._exit(object())
except TypeError:
    pass
else:
    raise SystemExit("os._exit(object()) did not raise")

# `os.abort` never returns either.  A process killed by SIGABRT reports a
# status that is not a clean exit; Windows spells it as a large unsigned code
# rather than a negative signal number, so the assertion is only that the
# child neither exited 0 nor reached the line after the call.
#
# The child suppresses the Windows crash dialog first, the way
# `test.support.suppress_msvcrt_asserts` does: an unattended abort otherwise
# waits for a person to dismiss it.
child = subprocess.run(
    [
        sys.executable,
        "-c",
        'import os, sys;'
        ' crt = __import__("msvcrt") if sys.platform == "win32" else None;'
        ' crt and crt.SetErrorMode(crt.SEM_FAILCRITICALERRORS'
        ' | crt.SEM_NOGPFAULTERRORBOX);'
        ' os.abort(); sys.stdout.write("after"); sys.stdout.flush()',
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=60,
    check=False,
)
assert child.returncode != 0, (child.returncode, child.stdout)
assert b"after" not in child.stdout, child.stdout

print("OK")
