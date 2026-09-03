"""`__format__` dispatch passes objects through, it does not rebuild them.

`format` (descroperation.py) resolves `__format__` on the type and calls it
with the spec the caller supplied, then returns what it got back:

    w_res = space.get_and_call_function(w_descr, w_obj, w_format_spec)
    if not space.isinstance_w(w_res, space.w_unicode): raise ...
    return w_res

Neither end is read out and rebuilt, so all three of the assertions below hold.
Reading the spec into a buffer and building a fresh `str` from it — at either
end — passes every equality test and fails all three of these.
"""


class SpecEcho:
    def __format__(self, spec):
        return "same" if spec is SPEC else "copy"


class Fixed:
    def __format__(self, spec):
        return RESULT


class MyStr(str):
    pass


class SubclassResult:
    def __format__(self, spec):
        return MyStr("x")


SPEC = ">4"
RESULT = "res"


def test_spec_reaches_dunder_as_the_same_object():
    assert format(SpecEcho(), SPEC) == "same"


def test_spec_survives_an_fstring_replacement_field():
    # A spec that is a single replacement field formats the operand with an
    # empty spec first; `format(s, "")` on an exact `str` hands back `s`, so
    # the object reaching `__format__` is still the one that was bound.
    assert f"{SpecEcho():{SPEC}}" == "same"


def test_result_is_the_object_the_dunder_returned():
    assert format(Fixed(), "") is RESULT
    assert f"{Fixed()}" is RESULT


def test_str_subclass_result_keeps_its_type():
    assert type(format(SubclassResult(), "")) is MyStr


def test_empty_spec_arrives_as_a_string():
    # FORMAT_SIMPLE carries no spec operand; the dunder's parameter is still a
    # `str`, so concatenating it is not a TypeError.
    class Concat:
        def __format__(self, spec):
            return "<" + spec + ">"

    assert f"{Concat()}" == "<>"
    assert f"{Concat():x}" == "<x>"


def test_non_str_spec_is_a_type_error():
    # The one place the two upstreams disagree: `format()` validates that
    # format_spec is a str, while pypy hands whatever it was given straight to
    # `__format__` and this raises nothing there.  The spec follows CPython, so
    # the TypeError is the behaviour to keep -- this asserts the check is still
    # applied, and that the dispatch below it did not start seeing the spec as
    # empty instead.
    assert format(SpecEcho(), "") == "copy"
    try:
        format(SpecEcho(), 34)
    except TypeError as exc:
        assert "must be str" in str(exc)
    else:
        raise AssertionError("format() accepted a non-str spec")
