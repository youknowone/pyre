//! The exception instance's slots, the two error indicators and the
//! `ImportError` builders.
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
import cpyext_exceptions as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


class MyImportError(ImportError):
    pass


class Strict(ImportError):
    def __init__(self, msg):
        ImportError.__init__(self, msg)


class Blows(ImportError):
    def __init__(self, *a, **k):
        raise RuntimeError('ctor blew up')


def a_traceback():
    try:
        raise ValueError('tb-source')
    except ValueError as e:
        return e.__traceback__


TB = a_traceback()

# ── PyException_* ──────────────────────────────────────────────────────

e = ValueError('boom')
# An empty slot is NULL rather than `None`, which the fixture spells `None`.
eq('traceback empty', m.exc_get_traceback(e), None)
eq('traceback set', m.exc_set_traceback(e, TB), (TB, True))
eq('traceback cleared', m.exc_set_traceback(e, None), (None, True))
eq('traceback refused', m.exc_set_traceback(e, 42),
   ('TypeError', '__traceback__ must be a traceback or None'))

e = ValueError('boom')
eq('cause empty', m.exc_get_cause(e), None)
cause = KeyError('the-cause')
# Writing the cause raises `__suppress_context__`, whatever the cause is.
eq('cause set', m.exc_set_cause(e, cause), (cause, True))
eq('cause cleared', m.exc_set_cause(e, None), (None, True))

e = ValueError('boom')
eq('context empty', m.exc_get_context(e), None)
context = KeyError('the-context')
# Writing the context leaves `__suppress_context__` alone.
eq('context set', m.exc_set_context(e, context), (context, False))
eq('context cleared', m.exc_set_context(e, None), (None, False))
eq('context suppress untouched', e.__suppress_context__, False)

eq('args', m.exc_get_args(ValueError('a', 'b')), ('a', 'b'))
eq('args empty', m.exc_get_args(ValueError()), ())
e = ValueError('boom')
eq('args set', m.exc_set_args(e, (1, 2)), ((1, 2), (1, 2)))
# The setter increfs rather than stealing, so the caller's tuple survives it.
held = (5, 6)
eq('args borrowed', m.exc_set_args(e, held), (held, held))

eq('classify class', m.exc_classify(ValueError), (True, False, 'ValueError', type))
eq('classify subclass', m.exc_classify(MyImportError),
   (True, False, 'MyImportError', type))
eq('classify instance', m.exc_classify(e), (False, True, '', ValueError))
eq('classify int', m.exc_classify(3), (False, False, '', int))
eq('classify int class', m.exc_classify(int), (False, False, '', type))
eq('classify base', m.exc_classify(BaseException), (True, False, 'BaseException', type))

# ── the raised indicator ───────────────────────────────────────────────

before, taken, after = m.raised_round_trip(ValueError, 'boom')
eq('round trip before', before, ValueError)
eq('round trip taken', (type(taken), str(taken)), (ValueError, 'boom'))
eq('round trip after', after, None)
eq('when clear', m.raised_when_clear(), (None, None))

k = KeyError('k')
eq('set steals', m.raised_set(k), (KeyError, k, True))
# The displaced exception loses exactly the reference the indicator held.
eq('set NULL clears', m.raised_set_null(KeyError('k')), (None, 1, True))
second = TypeError('second')
eq('replace releases', m.raised_replace(ValueError('first'), second), (1, second))

e = ValueError('with-tb')
e.__traceback__ = TB
eq('fetch triple', m.fetch_triple(e), (ValueError, e, TB, None))
eq('fetch triple clear', m.fetch_triple(None), (None, None, None, None))

# Fetching a triple and handing it straight back is what an extension does
# around work of its own, so the pair may lose nothing -- the traceback least
# of all, since only the triple carries it between the two calls.
eq('fetch and restore', m.fetch_and_restore(e), (e, TB, True))
bare = KeyError('bare')
eq('fetch and restore no traceback', m.fetch_and_restore(bare), (bare, None, True))

# `PyErr_SetObject` chains onto the exception being handled the way a raise
# from inside an `except` block does, and `PyErr_Restore` does not chain.
raised, context = m.set_object_context(KeyError, 'boom')
eq('set_object context outside', context, None)
try:
    raise ValueError('OLD')
except ValueError as old:
    raised, context = m.set_object_context(KeyError, 'boom')
    eq('set_object context inside', context, old)
    eq('restore does not chain', m.fetch_and_restore(KeyError('later'))[0].__context__, None)

# ── the handled exception ──────────────────────────────────────────────

eq('handled outside', m.handled_get(), None)
eq('handled twice outside', m.handled_twice(), (None, True))
try:
    raise ValueError('in-except')
except ValueError as outer:
    eq('handled inside', m.handled_get(), outer)
    # Reading it twice neither clears it nor answers a different object.
    eq('handled twice inside', m.handled_twice(), (outer, True))
    try:
        raise KeyError('inner')
    except KeyError as inner:
        eq('handled nested', m.handled_get(), inner)
    eq('handled back outside', m.handled_get(), outer)
eq('handled after', m.handled_get(), None)

held = KeyError('handled')
# The setter borrows: it takes a reference of its own, so the slot survives
# the caller releasing the one it handed over.
eq('handled set borrows', m.handled_set(held), (held, True))
eq('handled set None', m.handled_set(None), (None, False))

# ── the handled triple ─────────────────────────────────────────────────

# Empty: the class and traceback slots receive `None` and the value slot NULL,
# which is what tells this apart from `sys.exc_info()`.
eq('exc_info empty', m.exc_info(), ((None, None, None), True))
eq('exc_info derived empty', m.exc_info_derived(), None)
try:
    raise ValueError('in-except')
except ValueError as outer:
    eq('exc_info inside', m.exc_info(),
       ((ValueError, outer, outer.__traceback__), False))
    eq('exc_info derived', m.exc_info_derived(), (True, True))
# Only the value is stored; the class and the traceback are released and
# recomputed from it on the way out.
value = ValueError('v2')
eq('exc_info set', m.exc_info_set(str, value, 'NOT_A_TB'), (ValueError, value, None))
eq('exc_info set None', m.exc_info_set(None, None, None), (None, None, None))

# ── ImportError ────────────────────────────────────────────────────────

# Always answers NULL, which is the third field of every row here.
eq('import basic', m.import_error('boom', 'mod', '/p/mod.py'),
   ('ImportError', 'boom', True, 'mod', '/p/mod.py', None))
eq('import bare', m.import_error('boom', None, None),
   ('ImportError', 'boom', True, None, None, None))
eq('import int msg', m.import_error(42, 'mod', None),
   ('ImportError', '42', True, 'mod', None, None))
eq('import no msg', m.import_error(None, 'mod', '/p'),
   ('TypeError', 'expected a message argument', True, '<absent>', '<absent>', '<absent>'))
eq('import subclass', m.import_error_subclass(MyImportError, 'boom', 'mod', '/p'),
   ('MyImportError', 'boom', True))
eq('import mnfe', m.import_error_subclass(ModuleNotFoundError, 'boom', 'mod', None),
   ('ModuleNotFoundError', 'boom', True))
eq('import not a subclass', m.import_error_subclass(ValueError, 'boom', None, None),
   ('TypeError', 'expected a subclass of ImportError', True))
eq('import not a class', m.import_error_subclass(42, 'boom', None, None),
   ('TypeError', 'issubclass() arg 1 must be a class', True))
# The three keywords reach the constructor, so one that does not take them
# refuses.  Only the class and the opening of the message are compared: how a
# binder names more than one unexpected keyword is its own question.
strict = m.import_error_subclass(Strict, 'boom', None, None)
eq('import strict ctor', (strict[0], strict[1].startswith('Strict.__init__() got'),
                          strict[2]), ('TypeError', True, True))
eq('import raising ctor', m.import_error_subclass(Blows, 'boom', None, None),
   ('RuntimeError', 'ctor blew up', True))
# Implicitly chained onto the exception being handled, and never onto a cause.
eq('import context outside', m.import_error_context('boom'), (None, None))
try:
    raise ValueError('OLD')
except ValueError as old:
    eq('import context inside', m.import_error_context('boom'), (old, None))

print('cpyext-exceptions-ok')
"#;

#[test]
fn the_exception_entry_points() {
    let fixtures = Fixtures::new("cpyext-exceptions");
    fixtures.compile("cpyext_exceptions");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-exceptions-ok");
}
