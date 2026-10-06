# CPython-suite gap: `test_exception_group` constructs with two positionals
# and never `ExceptionGroup(message=..., exceptions=...)`, and
# `test_basics_subgroup_split__bad_arg_type` lists a non-exception class among
# `bad_args` without asserting the TypeError text.
#
# parity-tests reason: `BaseExceptionGroup_new` takes two positional
# arguments; keywords are not bound, so the call is `__new__() takes exactly
# 2 arguments (0 given)`. `get_matcher_type` rejects a class that is not an
# exception type. `W_BaseExceptionGroup.descr_new` binds `w_message` and
# `w_exceptions` and leaves keyword rejection to `descr_init`.
# `get_condition_filter` takes any `callable`.
#
# pyre-check: pypy-diverges: `descr_new` is `interp2app` with `w_message` and
# `w_exceptions`, so pypy3 answers `BaseExceptionGroup.__init__() takes no
# keyword arguments`. `get_condition_filter` calls a plain class, so
# `eg.subgroup(str)` answers a group.
try:
    ExceptionGroup(message="m", exceptions=[ValueError()])
except TypeError as err:
    assert str(err) == (
        "BaseExceptionGroup.__new__() takes exactly 2 arguments (0 given)"
    ), err
else:
    raise AssertionError("expected TypeError")

try:
    ExceptionGroup(1, [ValueError()])
except TypeError as err:
    assert str(err) == (
        "BaseExceptionGroup.__new__() argument 1 must be str, not int"
    ), err
else:
    raise AssertionError("expected TypeError")

try:
    ExceptionGroup("m", [ValueError()]).subgroup(str)
except TypeError as err:
    assert str(err) == (
        "expected an exception type, a tuple of exception types, or a "
        "callable (other than a class)"
    ), err
else:
    raise AssertionError("expected TypeError")

print("OK")
