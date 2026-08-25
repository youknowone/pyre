# pyre-check: selfcheck
# `c_call` / `c_return` belong to the call the BYTECODE made, not to whatever
# the interpreter reaches on its way there.
#
# Four dispatchers carry the C-profile arm — `baseobjspace.py call_valuestack`,
# `pyopcode.py CALL_FUNCTION_KW` and `CALL_FUNCTION_EX`, and `callmethod.py
# CALL_METHOD_KW` — and the bytecode hands every one of them the executing
# frame.  The test they apply is `is_builtin_code(w_function)`, which looks
# through a `_Method` and answers False for everything that is not a `Function`
# carrying a `BuiltinCode`.  A class is not one, so `C(1)` reports nothing at
# all — and the `__new__` and `__init__` that `typeobject.py descr_call` then
# runs report nothing either, because it reaches them through the frameless
# `space.call_args` / `get_and_call_args`, which carry no arm.
#
# THE TYPE CALLS ARE THE TEST.  Both halves have to hold at once, and only one
# of them is about the arm being present:
#
#   * `str(1)`, `D()`, `int('10', base=16)`, `int(*a, **kw)` and `dict(a=1)`
#     are calls to a type.  Each runs a builtin `__new__` and usually a builtin
#     `__init__` underneath, and NONE of that is reportable.  Every one of them
#     is checked at zero rather than by a difference, because a dispatcher that
#     arms itself off `gettopframe_raw` instead of off the frame it was handed
#     reports all of them and there is no tail length that hides it.
#   * `len`, `str.__new__` called directly, `lst.append`, `'a-b'.split(sep=…)`
#     and `len(*[…])` are calls the bytecode itself makes to a builtin, one per
#     dispatcher above.  Each is owed exactly `REPEAT` events, so a fix that
#     silences the type calls by narrowing the arm fails here instead.
#
# `len(1)` closes the pair: `baseobjspace.py call_args_and_c_profile` runs
# `c_exception_trace` and re-raises, so a raising builtin owes `c_call` and no
# `c_return`.
#
# Every body is warmed past the loop threshold with nothing installed, so each
# measured loop has already been compiled once when the hook arms.  pyre
# declines a profiled frame at the portal (see
# `profile_hook_armed_before_a_hot_loop.py`), so the measured pass itself runs
# interpreted — which is the point: the answer must not depend on which of the
# two ran, and the warm pass is what makes that claim testable rather than
# assumed.
#
# Measured on cpython 3.14.6 and pypy3, which agree on every count below.
import sys

REPEAT = 5
WARM = 5000  # past the loop threshold (1039) several times over

WARM_ITER = [0] * WARM
TEST_ITER = [0] * REPEAT


class C:
    def __init__(self, x):
        self.x = x


class D:
    pass


def measure(body):
    counts = {}

    def hook(frame, event, arg):
        if event in ('c_call', 'c_return'):
            name = getattr(arg, '__name__', repr(arg))
            counts[(event, name)] = counts.get((event, name), 0) + 1

    sys.setprofile(hook)
    try:
        body(TEST_ITER)
    finally:
        sys.setprofile(None)
    # `sys.setprofile` is itself a builtin called from this frame, and the
    # arming call is inside the window it opens.
    counts.pop(('c_call', 'setprofile'), None)
    return counts


def type_calls(it):
    for _ in it:
        str(1)
        C(1)
        D()
        dict(a=1)
        int('10', base=16)
        int(*['10'], **{'base': 16})


def plain_builtin(it):
    for _ in it:
        len('a')


def static_new(it):
    for _ in it:
        str.__new__(str, 1)


def bound_method(it):
    lst = []
    for _ in it:
        lst.append(1)


def method_with_keyword(it):
    for _ in it:
        'a-b'.split(sep='-')


def builtin_through_star(it):
    argv = [['a', 'b']]
    for _ in it:
        len(*argv)


def raising_builtin(it):
    for _ in it:
        try:
            len(1)
        except TypeError:
            pass


def check(counts, expected, arm, failures):
    for key, want in sorted(expected.items()):
        got = counts.get(key, 0)
        if got != want:
            failures.append('%s: %s = %d, expected %d' % (arm, key, got, want))
    for key in sorted(counts):
        if key not in expected:
            failures.append(
                '%s: %s = %d, expected no event — a dispatch the bytecode '
                'never made reported one' % (arm, key, counts[key])
            )


def main():
    failures = []
    cases = [
        # A call to a type reports nothing, however builtin its `__new__` and
        # `__init__` turn out to be.
        ('type calls', type_calls, {}),
        ('len', plain_builtin, {('c_call', 'len'): REPEAT, ('c_return', 'len'): REPEAT}),
        # The same `__new__` the type calls above run internally — reportable
        # here only because the bytecode is the one calling it.
        (
            'str.__new__',
            static_new,
            {('c_call', '__new__'): REPEAT, ('c_return', '__new__'): REPEAT},
        ),
        (
            'bound method',
            bound_method,
            {('c_call', 'append'): REPEAT, ('c_return', 'append'): REPEAT},
        ),
        (
            'CALL_KW',
            method_with_keyword,
            {('c_call', 'split'): REPEAT, ('c_return', 'split'): REPEAT},
        ),
        (
            'CALL_FUNCTION_EX',
            builtin_through_star,
            {('c_call', 'len'): REPEAT, ('c_return', 'len'): REPEAT},
        ),
        # c_exception_trace re-raises, so the return event is never owed.
        ('raising builtin', raising_builtin, {('c_call', 'len'): REPEAT}),
    ]
    for arm, body, expected in cases:
        body(WARM_ITER)
        check(measure(body), expected, arm, failures)
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS c_call reports the bytecode-level builtin call and nothing else')
    return 0


sys.exit(main())
