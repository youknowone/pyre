# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=through_plain,through_paired,through_spare,through_nested
# A CALL that inlined a constructor and then aborted inside `__init__` handed
# the caller `None` instead of the instance.
#
# The walker recognises `C(...)` while decoding the CALL, allocates the
# instance itself and inlines only `__init__` as a user call
# (`try_walker_inline_type_call`).  When that sub-walk aborts on something it
# cannot record -- here the loop in `__init__`, which refuses the inline with
# `LoopBearingCalleeInlineUnsupported` -- the gh#467 recovery rebuilds the
# callee frame on the interpreter, runs it to its return, and resumes the
# caller past the CALL with that return value.  In a real call
# `type_descr_call_impl` sits between the two frames, and its tail is what
# discards `__init__`'s `None` and hands back `w_newobject`; the rebuild had no
# frame for that tail, so the caller read `__init__`'s `None` as the value of
# `C(...)`.  `ctor_continuation` plays the same tail for the blackhole resume.
#
# One loop per class, so each gets its own trace and its own abort.  The miss
# is a single event at the moment a trace aborts, and a fixture with one shape
# could go vacuous on a timing shift alone.
#
# `Nested` is the fourth shape rather than a fourth copy of the first: the
# constructor sits inside another constructor's `__init__`, which is where
# `configparser.RawConfigParser.__init__` met this and read `None` for
# `ConverterMapping(self)`.
#
# Measured before the fix, on dynasm: all four reported one `None` out of
# 20000, and the same run under `PYRE_NO_JIT=1` reported none.  The threshold
# is pinned because the default one does not compile these loops early enough
# to abort inside them.
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=100,function_threshold=100")
except ImportError:
    pass

N = 20000


class Plain:
    def __init__(self):
        for _ in range(5):
            pass


class Paired:
    def __init__(self):
        for _ in range(5):
            pass


class Spare:
    def __init__(self):
        for _ in range(5):
            pass


class Inner:
    def __init__(self, owner):
        for _ in range(5):
            pass


class Nested:
    def __init__(self):
        self.inner = Inner(self)


def make_plain():
    return Plain()


def make_paired():
    return Paired()


def make_spare():
    return Spare()


def through_plain():
    missing = 0
    for _ in range(N):
        if make_plain() is None:
            missing += 1
    return missing


def through_paired():
    missing = 0
    for _ in range(N):
        if make_paired() is None:
            missing += 1
    return missing


def through_spare():
    missing = 0
    for _ in range(N):
        if make_spare() is None:
            missing += 1
    return missing


def through_nested():
    missing = 0
    for _ in range(N):
        if Nested().inner is None:
            missing += 1
    return missing


def main():
    cases = [
        ("plain wrapper", through_plain),
        ("paired wrapper", through_paired),
        ("spare wrapper", through_spare),
        ("nested constructor", through_nested),
    ]

    failures = []
    for label, fn in cases:
        missing = fn()
        if missing:
            failures.append(f"{label}: {missing} of {N} constructor calls evaluated to None")

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS an aborted constructor inline still evaluates to the instance")
    return 0


sys.exit(main())
