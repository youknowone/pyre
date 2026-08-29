# pyre-check: pypy-diverges: pypy3's `array` has no `+`, `+=` or slice
# assignment of its own -- they fall through to the generic operand message --
# and its item converter states pypy's own sentence, so none of the wording
# below is expressible there.
#
# CPython-suite gap: `test_array` asserts the exception *type* for every one of
# these refusals and never the message, so a placeholder in place of the
# operand's type name reads as a pass.
#
# parity-tests reason: every one of `array`'s own refusals names the operand it
# read -- `array_concat`, `array_inplace_concat` and `array_ass_subscr` each
# interpolate `Py_TYPE(v)->tp_name`, and the item converters go through
# `PyNumber_Index`, which names the type it could not convert.  A message with
# a fixed placeholder where the name belongs tells a program nothing about what
# it passed, and one that merges two conditions reports the wrong refusal for
# an array of the wrong kind.
import array


def refusal(fn):
    try:
        fn()
    except BaseException as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    raise AssertionError("accepted")


def ints():
    return array.array("i", [1, 2, 3])


def doubles():
    return array.array("d", [1.0, 2.0])


BAD_ARGUMENT = "TypeError: bad argument type for built-in operation"

# `array_concat` names the operand; an array of another kind is a different
# refusal, and `PyErr_BadArgument` is what states it.
assert refusal(lambda: ints() + object()) == (
    'TypeError: can only append array (not "object") to array'
), refusal(lambda: ints() + object())
assert refusal(lambda: ints() + [1]) == (
    'TypeError: can only append array (not "list") to array'
), refusal(lambda: ints() + [1])
assert refusal(lambda: ints() + doubles()) == BAD_ARGUMENT, refusal(lambda: ints() + doubles())

# `array_inplace_concat` states its own sentence, and hands the same-kind test
# to `array_do_extend`, which states another.
assert refusal(lambda: ints().__iadd__(object())) == (
    'TypeError: can only extend array with array (not "object")'
), refusal(lambda: ints().__iadd__(object()))
assert refusal(lambda: ints().__iadd__(doubles())) == (
    "TypeError: can only extend with array of same kind"
), refusal(lambda: ints().__iadd__(doubles()))
assert refusal(lambda: ints().extend(doubles())) == (
    "TypeError: can only extend with array of same kind"
), refusal(lambda: ints().extend(doubles()))

# `array_ass_subscr` splits the same two ways.
assert refusal(lambda: ints().__setitem__(slice(0, 2), object())) == (
    'TypeError: can only assign array (not "object") to array slice'
), refusal(lambda: ints().__setitem__(slice(0, 2), object()))
assert refusal(lambda: ints().__setitem__(slice(0, 2), [1, 2])) == (
    'TypeError: can only assign array (not "list") to array slice'
), refusal(lambda: ints().__setitem__(slice(0, 2), [1, 2]))
assert refusal(lambda: ints().__setitem__(slice(0, 2), doubles())) == BAD_ARGUMENT

# `fromlist` takes a list and nothing else; `frombytes` names what it got.
assert refusal(lambda: ints().fromlist(1)) == "TypeError: arg must be list"
assert refusal(lambda: ints().fromlist((1, 2))) == "TypeError: arg must be list"
assert refusal(lambda: ints().frombytes("ab")) == (
    "TypeError: a bytes-like object is required, not 'str'"
), refusal(lambda: ints().frombytes("ab"))

# `sq_repeat` reads a count, so a non-count is refused as a sequence repeat
# rather than as a failed conversion.
assert refusal(lambda: ints() * "x") == (
    "TypeError: can't multiply sequence by non-int of type 'str'"
), refusal(lambda: ints() * "x")
assert refusal(lambda: ints() * 1.5) == (
    "TypeError: can't multiply sequence by non-int of type 'float'"
), refusal(lambda: ints() * 1.5)

# Every integer typecode converts its item through `PyNumber_Index`.
for code in "bBhHiIlLqQ":
    got = refusal(lambda c=code: array.array(c, [1.5]))
    assert got == "TypeError: 'float' object cannot be interpreted as an integer", (code, got)
    got = refusal(lambda c=code: array.array(c, [None]))
    assert got == "TypeError: 'NoneType' object cannot be interpreted as an integer", (code, got)

# `_array_reconstructor` reads three of its four arguments through a converter
# that names what it was handed.
reconstruct = array._array_reconstructor
assert refusal(lambda: reconstruct(array.array, "i", "x", b"")) == (
    "TypeError: 'str' object cannot be interpreted as an integer"
), refusal(lambda: reconstruct(array.array, "i", "x", b""))
assert refusal(lambda: reconstruct(array.array, "i", 0, 1)) == (
    "TypeError: fourth argument should be bytes, not int"
), refusal(lambda: reconstruct(array.array, "i", 0, 1))
assert refusal(lambda: reconstruct(array.array, b"i", 0, b"")) == (
    "TypeError: _array_reconstructor() argument 2 must be a unicode character, not bytes"
), refusal(lambda: reconstruct(array.array, b"i", 0, b""))
assert refusal(lambda: reconstruct(int, "i", 0, b"")) == (
    "TypeError: int is not a subtype of array.array"
), refusal(lambda: reconstruct(int, "i", 0, b""))
assert reconstruct(array.array, "b", 1, b"\x01\x02").tolist() == [1, 2]

print("OK")
