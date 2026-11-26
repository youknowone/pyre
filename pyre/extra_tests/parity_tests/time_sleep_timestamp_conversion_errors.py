# pyre-check: pypy-diverges: pypy3's `timestamp_w` reports "timestamp is nan"
# and "timestamp %R too large to convert to C _PyTime_t", and its unchecked
# `sec * SECS_TO_NS` wraps `2**63//10**9+1` into a negative nanosecond count
# that then raises ValueError "sleep length must be non-negative".
#
# CPython-suite gap: `test_time` only checks the overflow message through
# `_testinternalcapi._PyTime_FromSecondsObject`
# (`test_FromSecondsObject_float_overflow_message`), which pyre does not have.
#
# parity-tests reason: `time.sleep` converts its argument to an int64
# nanosecond count before the non-negative check.  So NaN is a ValueError and
# a value outside `_PyTime_t` is OverflowError, including an integer whose
# seconds fit in i64 but whose nanoseconds do not.
import time


def raises(exc, fn, *args, match=None):
    try:
        fn(*args)
    except exc as e:
        if match is not None:
            assert str(e) == match, (fn, args, str(e))
    else:
        raise AssertionError((fn, args, "did not raise", exc))


raises(ValueError, time.sleep, float("nan"), match="Invalid value NaN (not a number)")
raises(OverflowError, time.sleep, 2**80, match="timestamp out of range for C PyTime_t")
raises(OverflowError, time.sleep, -2**80, match="timestamp out of range for C PyTime_t")
raises(OverflowError, time.sleep, 1e300, match="timestamp out of range for C PyTime_t")
raises(OverflowError, time.sleep, -1e300, match="timestamp out of range for C PyTime_t")
raises(OverflowError, time.sleep, 1e20, match="timestamp out of range for C PyTime_t")
raises(
    OverflowError,
    time.sleep,
    2**63 // 10**9 + 1,
    match="timestamp out of range for C PyTime_t",
)
raises(ValueError, time.sleep, -1, match="sleep length must be non-negative")
assert time.sleep(0) is None

print("OK")
