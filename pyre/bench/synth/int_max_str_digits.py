# PEP 674 integer string-conversion length limit: sys.get/set_int_max_str_digits
# with the default 4300, enforced on str()/repr()/f-string of a large int and on
# int() parsing of a long decimal literal, bypassed for power-of-two bases, and
# disabled by a limit of 0. (Error text and whether an explicit format code
# enforces the limit both diverge between the reference interpreters, so only the
# exception type is printed here.)
import sys


def main():
    print(sys.get_int_max_str_digits())
    print(len(str(10 ** 4299)))  # exactly 4300 digits -> allowed

    for label, fn in [
        ("over", lambda: str(10 ** 4300)),
        ("neg", lambda: str(-(10 ** 5000))),
        ("repr", lambda: repr(10 ** 5000)),
        ("fstr", lambda: f"{10 ** 5000}"),
        ("parse", lambda: int("1" * 5000)),
    ]:
        try:
            fn()
            print(label, "ok")
        except ValueError:
            print(label, "ValueError")

    sys.set_int_max_str_digits(6000)
    print(sys.get_int_max_str_digits())
    print(len(str(10 ** 5000)))  # 5001 digits <= 6000 -> allowed

    try:
        sys.set_int_max_str_digits(100)  # below threshold -> rejected
        print("set100 ok")
    except ValueError:
        print("set100 ValueError")

    sys.set_int_max_str_digits(0)  # 0 disables the limit
    print(len(str(10 ** 5000)))

    print(int("f" * 5000, 16) > 0)  # base 16 bypasses the limit


main()
