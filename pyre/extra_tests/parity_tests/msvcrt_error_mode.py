# pyre-check: platforms=win32
"""The process error mode is a global the caller is expected to put back.

`SetErrorMode` reports the mode it replaced rather than the one it installed,
which is the whole protocol for restoring it: read the old value out of the
setter's return.  That works for a caller that is changing the mode, and not
for one that only wants to know what it is -- asking with `SetErrorMode` means
setting it, and setting it is what such a caller is trying not to do.
`GetErrorMode` is the read that does not write, and `faulthandler` and
`ctypes`' crash-dialog suppression both open with it.
"""

import msvcrt

SEM_FAILCRITICALERRORS = 0x0001
SEM_NOGPFAULTERRORBOX = 0x0002

original = msvcrt.GetErrorMode()
assert isinstance(original, int), type(original)

try:
    # The setter answers with what was there, and the getter then answers with
    # what was installed -- so the two agree about the same moment.
    previous = msvcrt.SetErrorMode(SEM_FAILCRITICALERRORS)
    assert previous == original, (previous, original)
    assert msvcrt.GetErrorMode() == SEM_FAILCRITICALERRORS, msvcrt.GetErrorMode()

    combined = SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
    previous = msvcrt.SetErrorMode(combined)
    assert previous == SEM_FAILCRITICALERRORS, previous
    assert msvcrt.GetErrorMode() == combined, msvcrt.GetErrorMode()

    # Reading twice does not change it, which is the point of the call.
    assert msvcrt.GetErrorMode() == msvcrt.GetErrorMode() == combined
finally:
    msvcrt.SetErrorMode(original)

assert msvcrt.GetErrorMode() == original, (msvcrt.GetErrorMode(), original)

# It takes no arguments.
try:
    msvcrt.GetErrorMode(0)
except TypeError:
    pass
else:
    raise AssertionError("GetErrorMode takes no arguments")

print("OK")
