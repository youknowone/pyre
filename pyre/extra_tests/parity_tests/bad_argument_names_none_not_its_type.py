# pyre-check: pypy-diverges: pins 3.14's argument-clinic rejection wording;
# pypy3 words each of these differently ("expected str, got NoneType object",
# "argument 1 must be str, not <class 'NoneType'>").
#
# CPython-suite gap: the suite checks that each of these calls raises TypeError
# and, where it looks at the message at all, matches a substring that stops
# before the type name.  Nothing there asserts which spelling the rejected
# argument gets, so an implementation that always prints the type reads as
# correct.
#
# parity-tests reason: `_PyArg_BadArgument` renders a rejected `None` as `None`
# and every other value as its type name, so the two spellings are not
# interchangeable, and the choice is per call site: `str()` and `bytes()` reject
# their `encoding` through a different check and do print `NoneType`.  A shared
# helper that always prints the type, or one applied to every site at once,
# breaks one group or the other, and only the message shows it.
def show(fn):
    try:
        fn()
    except TypeError as exc:
        return str(exc)
    raise AssertionError('expected TypeError')


def clinic_rejections_spell_none_as_none():
    cases = [
        (lambda: BaseExceptionGroup(None, [ValueError()]),
         'BaseExceptionGroup.__new__() argument 1 must be str, not None'),
        (lambda: UnicodeEncodeError(None, 'a', 0, 1, 'r'),
         'argument 1 must be str, not None'),
        (lambda: UnicodeDecodeError(None, b'a', 0, 1, 'r'),
         'argument 1 must be str, not None'),
        (lambda: type(None, (), {}),
         'type.__new__() argument 1 must be str, not None'),
        (lambda: open('x', None),
         "open() argument 'mode' must be str, not None"),
        (lambda: str.maketrans('ab', None),
         'maketrans() argument 2 must be str, not None'),
        (lambda: 'a'.encode(None),
         "encode() argument 'encoding' must be str, not None"),
        (lambda: b'a'.decode(None),
         "decode() argument 'encoding' must be str, not None"),
        (lambda: ValueError().add_note(None),
         'add_note() argument must be str, not None'),
        (lambda: type(__import__('sys'))(None),
         "module() argument 'name' must be str, not None"),
        (lambda: bytes('a', None),
         "bytes() argument 'encoding' must be str, not None"),
    ]
    for fn, expected in cases:
        got = show(fn)
        assert got == expected, (got, expected)


def a_rejected_value_that_is_not_none_names_its_type():
    cases = [
        (lambda: BaseExceptionGroup(1, [ValueError()]),
         'BaseExceptionGroup.__new__() argument 1 must be str, not int'),
        (lambda: open('x', 1),
         "open() argument 'mode' must be str, not int"),
        (lambda: 'a'.encode(1),
         "encode() argument 'encoding' must be str, not int"),
    ]
    for fn, expected in cases:
        got = show(fn)
        assert got == expected, (got, expected)


def the_str_constructor_names_the_type_of_a_none_encoding():
    cases = [
        (lambda: str(b'', None),
         "str() argument 'encoding' must be str, not NoneType"),
        (lambda: str(b'', 'utf-8', None),
         "str() argument 'errors' must be str, not NoneType"),
    ]
    for fn, expected in cases:
        got = show(fn)
        assert got == expected, (got, expected)


clinic_rejections_spell_none_as_none()
a_rejected_value_that_is_not_none_names_its_type()
the_str_constructor_names_the_type_of_a_none_encoding()
print('OK')
