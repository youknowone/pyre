# pyre-check: gate=1
# pyre-check: platforms=darwin,linux
# CPython-suite gap: test_signal never asks for a signal number above the
# platform's range, so nothing in the suite reads the bound back.
# parity-tests reason: `NSIG` was a single 64 on every unix, so `getsignal`
# and `strsignal` accepted numbers the platform has no signal for, and
# `valid_signals()` answered with a bare int for a signal with no name.

# pyre-check: pypy-diverges: `interp_signal.py strsignal` spells its own bound
# as `signalnum > NSIG`, which admits `NSIG` itself and answers
# `'Unknown signal: 32'`; 3.14 reports the range as `[1; NSIG - 1]` and
# raises.

# `NSIG` is what `Py_NSIG` names: one past the highest signal the platform
# has.  It is 32 on darwin and 65 under glibc, so every assertion below is
# written against the value rather than a literal.  Windows counts its own
# 23 but has neither `SIGUSR1` nor the POSIX delivery model, so it is out of
# scope here.

import signal

NSIG = signal.NSIG
assert NSIG > 1, NSIG

# The half-open bound: `1` is a signal, `NSIG` is one past the last.
assert signal.getsignal(1) is not None
signal.strsignal(1)

for out_of_range in (0, NSIG, NSIG + 1, NSIG + 8):
    for name in ("getsignal", "strsignal"):
        try:
            getattr(signal, name)(out_of_range)
        except ValueError as e:
            assert str(e) == "signal number out of range", (name, out_of_range, e)
        else:
            raise AssertionError(f"{name}({out_of_range}) did not raise")

# `valid_signals()` sweeps `sigismember` over the same half-open range, so
# nothing it reports can sit at or past the bound.  Membership is not
# enum-membership: glibc's realtime signals have no `Signals` name and come
# back as plain ints on every interpreter.
valid = signal.valid_signals()
assert valid, valid
for sig in valid:
    assert isinstance(sig, int), (sig, type(sig))
    assert 1 <= sig < NSIG, sig

# The pending-signal bits are still delivered after the bound moved.
received = []
previous = signal.signal(signal.SIGUSR1, lambda n, f: received.append(n))
try:
    signal.raise_signal(signal.SIGUSR1)
finally:
    signal.signal(signal.SIGUSR1, previous)
assert received == [signal.SIGUSR1], received

print("OK")
