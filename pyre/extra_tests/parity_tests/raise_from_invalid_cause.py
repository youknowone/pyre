# CPython-suite gap: test_raise.TestCause.test_invalid_cause raises
# `IndexError from 5`, whose class has no observable constructor, so no test
# sees whether the raised class is instantiated before the cause is rejected,
# and none raises a non-exception value with an invalid cause.
# parity-tests reason: pyre validated the cause where it normalized it, one
# step ahead of `set_cause`, and the three shapes below are what that reorder
# is observable as.

"""`raise X from Y` validates Y only once X has been normalized.

`pyopcode.py:757-775 RAISE_VARARGS` instantiates a class cause, pops the
raised value, instantiates that too, normalizes it, and only then calls
`error.py:376-385 set_cause`, which is where a cause that does not derive from
`BaseException` becomes a TypeError.  Three things follow, and pyre answered
none of them: the raised class runs its constructor even when the cause is
invalid; a raised value that is not an exception reports its own TypeError
rather than the cause's; and a cause constructor that raises propagates that
exception instead of being read as an absent cause.

The raise runs hot so the JIT residual (`bh_normalize_raise_varargs_with_frame`)
and the tracer answer alongside the plain interpreter.
"""

ROUNDS = 400


class Raised(Exception):
    """Records that the raised class was instantiated."""

    constructed = 0

    def __init__(self, *args):
        Raised.constructed += 1
        super().__init__(*args)


class CauseThatRaises(Exception):
    def __init__(self, *args):
        raise ValueError("cause constructor")


def invalid_cause_still_builds_the_raised_class():
    """`raise Raised from 42` runs `Raised()`, then rejects the cause."""
    before = Raised.constructed
    try:
        raise Raised from 42
    except TypeError as error:
        message = str(error)
    else:
        raise AssertionError("an invalid cause must raise TypeError")
    assert Raised.constructed == before + 1, (
        f"the raised class was not instantiated ({before} -> {Raised.constructed})"
    )
    assert "cause" in message, message


def a_bad_raised_value_reports_its_own_error():
    """`raise 5 from 42` is the value's TypeError, not the cause's."""
    try:
        raise 5 from 42
    except TypeError as error:
        message = str(error)
    else:
        raise AssertionError("raising a non-exception must raise TypeError")
    assert "cause" not in message, message


def a_cause_constructor_error_propagates():
    """`raise Raised from CauseThatRaises` is the constructor's ValueError.

    Whether `Raised()` has run by then is not asserted: `pyopcode.py:757-760`
    instantiates the cause before popping the value, while the reference does
    it the other way round, so the two disagree on that and only that.
    """
    try:
        raise Raised from CauseThatRaises
    except ValueError as error:
        assert str(error) == "cause constructor", error
    except BaseException as other:
        raise AssertionError(
            f"the cause constructor's error was swallowed: {type(other).__name__}"
        ) from None
    else:
        raise AssertionError("the cause constructor must propagate")


def valid_causes_are_unchanged():
    """The shapes that already worked keep working."""
    cause = KeyError("k")
    try:
        raise Raised("v") from cause
    except Raised as error:
        assert error.__cause__ is cause
        assert error.__suppress_context__ is True

    try:
        raise Raised("v") from None
    except Raised as error:
        assert error.__cause__ is None
        assert error.__suppress_context__ is True

    try:
        raise Raised("v") from KeyError
    except Raised as error:
        assert isinstance(error.__cause__, KeyError)


for _ in range(ROUNDS):
    invalid_cause_still_builds_the_raised_class()
    a_bad_raised_value_reports_its_own_error()
    a_cause_constructor_error_propagates()
    valid_causes_are_unchanged()

print("OK")
