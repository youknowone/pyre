//! The warnings entry points and the `PyExc_*` mirrors an extension links
//! against.
//!
//! Every expectation was taken from CPython 3.14.6 running this same script
//! against this same fixture.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import warnings

import cpyext_warnings as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


def caught(fn):
    """The entry point's answer and the warnings it left behind."""
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always')
        answer = fn()
    return answer, [(type(w.message).__name__, str(w.message)) for w in log]


def located(fn):
    """[`caught`] with each warning's location."""
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always')
        answer = fn()
    return answer, [(type(w.message).__name__, str(w.message), w.filename, w.lineno)
                    for w in log]


def sourced(fn):
    """[`caught`] with each warning's `source`."""
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always')
        answer = fn()
    return answer, [(type(w.message).__name__, str(w.message), w.source) for w in log]


def under(action, fn):
    with warnings.catch_warnings():
        warnings.simplefilter(action)
        return fn()


class Unrepresentable:
    def __repr__(self):
        raise RuntimeError('repr blew up')


def emit(level):
    return m.warn_ex(UserWarning, 'at-%d' % level, level)


EMIT_LINE = emit.__code__.co_firstlineno + 1
# `-c` names the script `<string>`; a file names its path.
HERE = emit.__code__.co_filename


def outer():
    return emit(2)


OUTER_LINE = outer.__code__.co_firstlineno + 1

# ── PyErr_WarnEx ───────────────────────────────────────────────────────

eq('plain', caught(lambda: m.warn_ex(UserWarning, 'plain', 1)),
   ((0, None), [('UserWarning', 'plain')]))
# A NULL category is `RuntimeWarning`, not `UserWarning`.
eq('null category', caught(lambda: m.warn_ex(None, 'null-cat', 1)),
   ((0, None), [('RuntimeWarning', 'null-cat')]))
eq('deprecation', caught(lambda: m.warn_ex(DeprecationWarning, 'dep', 1)),
   ((0, None), [('DeprecationWarning', 'dep')]))
eq('empty message', caught(lambda: m.warn_ex(UserWarning, '', 1)),
   ((0, None), [('UserWarning', '')]))
# The category is never checked: it is called with the message, so a class
# that is not a `Warning` is emitted and one that is not callable refuses.
eq('not a warning', caught(lambda: m.warn_ex(ValueError, 'not-warning', 1)),
   ((0, None), [('ValueError', 'not-warning')]))
eq('not a class', caught(lambda: m.warn_ex(42, 'not-class', 1)),
   ((-1, ('TypeError', "'int' object is not callable")), []))
eq('instance category', caught(lambda: m.warn_ex(UserWarning('i'), 'inst', 1)),
   ((-1, ('TypeError', "'UserWarning' object is not callable")), []))
# The filter decides the answer: `error` raises out of the entry point.
eq('error filter',
   caught(lambda: under('error', lambda: m.warn_ex(UserWarning, 'as-error', 1))),
   ((-1, ('UserWarning', 'as-error')), []))
eq('ignore filter',
   caught(lambda: under('ignore', lambda: m.warn_ex(UserWarning, 'ignored', 1))),
   ((0, None), []))
# A message that is not valid UTF-8 is refused, with the decode error the
# `utf-8` codec would have raised for the same bytes.
eq('bad utf8 start', m.warn_ex_bytes(UserWarning, b'bad\xff', 1),
   (-1, ('UnicodeDecodeError',
         "'utf-8' codec can't decode byte 0xff in position 3: invalid start byte")))
eq('bad utf8 continuation', m.warn_ex_bytes(UserWarning, b'\xc3(', 1),
   (-1, ('UnicodeDecodeError',
         "'utf-8' codec can't decode byte 0xc3 in position 0: invalid continuation byte")))
eq('bad utf8 truncated', m.warn_ex_bytes(UserWarning, b'ok\xc3', 1),
   (-1, ('UnicodeDecodeError',
         "'utf-8' codec can't decode byte 0xc3 in position 2: unexpected end of data")))

# `PyErr_Warn` is `PyErr_WarnEx` with a stack level of 1.
eq('macro plain', caught(lambda: m.warn_macro(UserWarning, 'macro')),
   ((0, None), [('UserWarning', 'macro')]))
eq('macro null', caught(lambda: m.warn_macro(None, 'macro-null')),
   ((0, None), [('RuntimeWarning', 'macro-null')]))

# ── where the stack level puts the warning ─────────────────────────────

eq('level 1', located(lambda: emit(1)),
   ((0, None), [('UserWarning', 'at-1', HERE, EMIT_LINE)]))
# A level below 1 behaves as 1.
eq('level 0', located(lambda: emit(0)),
   ((0, None), [('UserWarning', 'at-0', HERE, EMIT_LINE)]))
eq('level -5', located(lambda: emit(-5)),
   ((0, None), [('UserWarning', 'at--5', HERE, EMIT_LINE)]))
eq('level 2', located(outer),
   ((0, None), [('UserWarning', 'at-2', HERE, OUTER_LINE)]))
# Walking off the top of the stack names `sys` rather than a frame.
eq('level 1000', located(lambda: emit(1000)),
   ((0, None), [('UserWarning', 'at-1000', '<sys>', 0)]))

# ── PyErr_WarnFormat ───────────────────────────────────────────────────

eq('format plain', caught(lambda: m.warn_format(UserWarning, 1, [1, 2])),
   ((0, None), [('UserWarning', 's=abc d=42 o=[1, 2]')]))
eq('format null cat', caught(lambda: m.warn_format(None, 1, {'k': 1})),
   ((0, None), [('RuntimeWarning', "s=abc d=42 o={'k': 1}")]))
# `%R` runs the object's `__repr__`, so the message can fail to be built.
eq('format bad repr', caught(lambda: m.warn_format(UserWarning, 1, Unrepresentable())),
   ((-1, ('RuntimeError', 'repr blew up')), []))
eq('format error filter',
   caught(lambda: under('error', lambda: m.warn_format(UserWarning, 1, None))),
   ((-1, ('UserWarning', 's=abc d=42 o=None')), []))

# ── PyErr_ResourceWarning ──────────────────────────────────────────────

eq('resource none', sourced(lambda: m.resource_warning(None, 1)),
   ((0, None), [('ResourceWarning', 'unclosed sock at 4', None)]))
SOURCE = [1, 2, 3]
eq('resource source', sourced(lambda: m.resource_warning(SOURCE, 1)),
   ((0, None), [('ResourceWarning', 'unclosed sock at 4', SOURCE)]))

# ── PyErr_WarnExplicit ─────────────────────────────────────────────────

eq('explicit plain',
   located(lambda: m.warn_explicit(UserWarning, 'ex-plain', '/a/b.py', 11, 'a.b', None)),
   ((0, None), [('UserWarning', 'ex-plain', '/a/b.py', 11)]))
eq('explicit null cat',
   located(lambda: m.warn_explicit(None, 'ex-null', '/a/b.py', 12, 'a.b', None)),
   ((0, None), [('RuntimeWarning', 'ex-null', '/a/b.py', 12)]))
eq('explicit not warning',
   located(lambda: m.warn_explicit(ValueError, 'ex-vw', '/a/b.py', 13, 'a.b', None)),
   ((0, None), [('ValueError', 'ex-vw', '/a/b.py', 13)]))
# A NULL module is derived from the filename: `.py` stripped, and `<unknown>`
# for a filename that is nothing but the suffix.
eq('explicit null module',
   located(lambda: m.warn_explicit(UserWarning, 'ex-nomod', '/a/b.py', 14, None, None)),
   ((0, None), [('UserWarning', 'ex-nomod', '/a/b.py', 14)]))
eq('explicit module filter',
   caught(lambda: under('ignore', lambda: m.warn_explicit(
       UserWarning, 'ex-mod', '/a/b.py', 15, None, None))),
   ((0, None), []))
eq('explicit error filter',
   caught(lambda: under('error', lambda: m.warn_explicit(
       UserWarning, 'ex-error', '/a/b.py', 16, 'a.b', None))),
   ((-1, ('UserWarning', 'ex-error')), []))
# A registry that is neither a dict nor None is refused before anything else.
eq('explicit registry list',
   caught(lambda: m.warn_explicit(UserWarning, 'ex-bad', '/a/b.py', 17, 'a.b', [1])),
   ((-1, ('TypeError', "'registry' must be a dict or None")), []))

# Under `default` the registry remembers what it has already shown.
REGISTRY = {}


def once_through_registry():
    first = m.warn_explicit(UserWarning, 'ex-reg', '/a/b.py', 18, 'a.b', REGISTRY)
    second = m.warn_explicit(UserWarning, 'ex-reg', '/a/b.py', 18, 'a.b', REGISTRY)
    return first, second


eq('explicit registry',
   caught(lambda: under('default', once_through_registry)),
   (((0, None), (0, None)), [('UserWarning', 'ex-reg')]))
eq('explicit registry key',
   sorted(key for key in REGISTRY if key != 'version'),
   [('ex-reg', UserWarning, 18)])

# ── PyErr_WarnExplicitObject ───────────────────────────────────────────

eq('object plain',
   located(lambda: m.warn_explicit_object(
       UserWarning, 'obj-plain', '/a/c.py', 21, 'a.c', None)),
   ((0, None), [('UserWarning', 'obj-plain', '/a/c.py', 21)]))
# A message that is already a warning instance is emitted as it is, and its
# own class wins over the one passed.
eq('object message instance',
   located(lambda: m.warn_explicit_object(
       DeprecationWarning, UserWarning('obj-inst'), '/a/c.py', 22, 'a.c', None)),
   ((0, None), [('UserWarning', 'obj-inst', '/a/c.py', 22)]))
# `None` for the module is not "derive one": the warning is dropped.
eq('object module none',
   located(lambda: m.warn_explicit_object(
       UserWarning, 'obj-none', '/a/c.py', 23, None, None)),
   ((0, None), []))
# The filename is never type-checked; it reaches the record as it is.
eq('object filename int',
   located(lambda: m.warn_explicit_object(
       UserWarning, 'obj-int', 42, 24, 'a.c', None)),
   ((0, None), [('UserWarning', 'obj-int', 42, 24)]))
eq('object registry list',
   caught(lambda: m.warn_explicit_object(
       UserWarning, 'obj-bad', '/a/c.py', 25, 'a.c', [1])),
   ((-1, ('TypeError', "'registry' must be a dict or None")), []))
# Every optional argument NULL: the category defaults and the module is
# derived from the filename.
eq('object all null',
   located(lambda: m.warn_explicit_object_null('obj-null', '/a/c.py', 26)),
   ((0, None), [('RuntimeWarning', 'obj-null', '/a/c.py', 26)]))

# ── PyErr_WarnExplicitFormat ───────────────────────────────────────────

eq('exfmt plain',
   located(lambda: m.warn_explicit_format(UserWarning, '/a/d.py', 31, 'a.d', None)),
   ((0, None), [('UserWarning', 's=xyz d=-7', '/a/d.py', 31)]))
eq('exfmt null module',
   located(lambda: m.warn_explicit_format(UserWarning, '/a/d.py', 32, None, None)),
   ((0, None), [('UserWarning', 's=xyz d=-7', '/a/d.py', 32)]))

# ── the exception mirrors an extension links against ───────────────────

# Every mirror resolves, and the only two that answer to another class's name
# are the aliases `OSError` kept from before 3.3.
eq('mirrors', [pair for pair in m.mirrors() if pair[0] != pair[1]],
   [('EnvironmentError', 'OSError'), ('IOError', 'OSError')])

print('cpyext-warnings-ok')
"#;

#[test]
fn the_warning_entry_points() {
    let fixtures = Fixtures::new("cpyext-warnings");
    fixtures.compile("cpyext_warnings");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-warnings-ok");
}
