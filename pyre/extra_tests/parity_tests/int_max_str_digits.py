"""Python 3.14 non-binary int/string conversion limit."""

import sys

old = sys.get_int_max_str_digits()
try:
    sys.set_int_max_str_digits(640)
    assert sys.get_int_max_str_digits() == 640
    int("1" * 640)
    str(10**639)
    for operation in (lambda: int("1" * 641), lambda: str(10**640)):
        try:
            operation()
        except ValueError as exc:
            assert "conversion" in str(exc)
        else:
            raise AssertionError("decimal conversion limit was not enforced")
    int("1" * 10_000, 16)
    sys.set_int_max_str_digits(0)
    int("1" * 1000)
finally:
    sys.set_int_max_str_digits(old)

print("OK")
