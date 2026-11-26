# pyre-check: pypy-diverges: pypy3 3.11 reports "integer division or modulo
# by zero" for `2**100 // 0` and "long division or modulo by zero" for
# `2**100 % 0`; this pins CPython 3.15's "division by zero".
#
# CPython-suite gap: longobject tests do not compile long `//` / `%` with a
# zero divisor after a hot nonzero warmup and then inspect the message.
# parity-tests reason: GuardNoException after the bigint residual must
# resume in the helper catch_exception so Python sees "division by zero".

big = 2 ** 100


def floor_int(d):
    return big // d


def mod_int(d):
    return big % d


def floor_long(d):
    return big // (d * big)


def main():
    bad = []
    for fn in (floor_int, mod_int, floor_long):
        for i in range(3000):
            d = 0 if i % 7 == 0 else (i | 1)
            if d == 0:
                try:
                    fn(d)
                except ZeroDivisionError as e:
                    if str(e) != "division by zero":
                        bad.append((fn.__name__, i, str(e)))
                        break
                else:
                    bad.append((fn.__name__, i, "did not raise"))
                    break
            else:
                try:
                    fn(d)
                except ZeroDivisionError as e:
                    bad.append((fn.__name__, i, "unexpected ZeroDivisionError: %s" % (e,)))
                    break
    assert not bad, bad
    print("OK")


main()
