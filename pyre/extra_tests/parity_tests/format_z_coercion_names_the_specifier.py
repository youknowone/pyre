# pyre-check: pypy-diverges: pypy3 states the sentence without naming a
# specifier, so the wording this pins is not expressible there.
#
# CPython-suite gap: `test_format` and `test_long` exercise `z` on the
# presentations that accept it, and the rejections they do assert are checked
# by exception type, not by wording.
#
# parity-tests reason: `z` asks for negative zero to be coerced, which only a
# floating presentation can do, so `formatter_unicode.c` refuses it once per
# specifier kind and names which one it was reading.  A runtime that drops the
# name answers with a sentence that fits neither.
def refusal(value, spec):
    try:
        format(value, spec)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("%r accepted %r" % (value, spec))


INTEGER = "Negative zero coercion (z) not allowed in integer format specifier"
STRING = "Negative zero coercion (z) not allowed in string format specifier"

assert refusal(1, "z") == INTEGER, refusal(1, "z")
assert refusal(1, "zd") == INTEGER, refusal(1, "zd")
# `bool` formats through the integer specifier, so it states the same one.
assert refusal(True, "z") == INTEGER, refusal(True, "z")
assert refusal("a", "z") == STRING, refusal("a", "z")
assert refusal("a", "zs") == STRING, refusal("a", "zs")

# A floating presentation is what `z` is for, so these are not refusals at all.
assert format(-0.0, "z") == "0.0", format(-0.0, "z")
assert format(-0.0, "zf") == "0.000000", format(-0.0, "zf")
assert format(1, "zf") == "1.000000", format(1, "zf")

print("OK")
