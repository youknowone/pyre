//! The interpreter state a call runs inside: the namespace a name falls back
//! to, the context variables, and reporting a failure the caller has no way to
//! hand back.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const BUILTINS_SCRIPT: &str = r#"
import builtins as builtins_module

import cpyext_context as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


# The namespace, not the module: what a name lookup falls through to.
namespace = m.builtins()
eq('the fallback namespace is a dict', type(namespace) is dict, True)
eq('and it holds the names', namespace['len'] is len, True)
eq('and it is the module namespace',
   namespace is builtins_module.__dict__, True)


# The running frame's `__builtins__` is what decides, so a call made under a
# namespace of its own reads that one.
private = {'__import__': __import__, 'marker': 'private'}
source = 'import cpyext_context as m\nseen = m.builtins()\n'
scope = {'__builtins__': private}
exec(compile(source, '<probe>', 'exec'), scope)
eq('a frame with a namespace of its own', scope['seen'] is private, True)
eq('and the marker only that one carries', scope['seen']['marker'], 'private')

# A frame that names the module rather than the namespace is answered with the
# module's namespace all the same.
scope = {'__builtins__': builtins_module}
exec(compile(source, '<probe>', 'exec'), scope)
eq('a frame that names the module',
   scope['seen'] is builtins_module.__dict__, True)

print('cpyext-builtins-ok')
"#;

const CONTEXTVARS_SCRIPT: &str = r#"
import contextvars

import cpyext_context as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


plain = m.var_new('plain')
eq('what a made variable is', type(plain) is contextvars.ContextVar, True)
eq('and the name it carries', plain.name, 'plain')

# No value in this context and no default is not a failure: the slot is left
# empty and the call still reports success.
eq('a variable with nothing behind it', m.var_get(plain), ('unset',))
eq('and reading it with a default', m.var_get(plain, 'fallback'),
   ('value', 'fallback'))

token = m.var_set(plain, 'first')
eq('what a set hands back', type(token) is contextvars.Token, True)
eq('the value it now holds', m.var_get(plain), ('value', 'first'))
eq('and Python reads the same', plain.get(), 'first')
# A default is only consulted when there is nothing to read.
eq('a default beside a value', m.var_get(plain, 'fallback'), ('value', 'first'))

eq('resetting it', m.var_reset(plain, token), None)
eq('leaves nothing behind it again', m.var_get(plain), ('unset',))

# The default the variable was made with, which every read falls back to.
defaulted = m.var_new('defaulted', 'built-in')
eq('a variable made with a default', m.var_get(defaulted), ('value', 'built-in'))
eq('and Python reads the same', defaulted.get(), 'built-in')
inner = m.var_set(defaulted, 'set')
eq('a value over the default', m.var_get(defaulted), ('value', 'set'))
m.var_reset(defaulted, inner)
eq('and the default again after the reset',
   m.var_get(defaulted), ('value', 'built-in'))

# A variable set in a context of its own is not set outside it.
copied = contextvars.copy_context()
eq('what a copied context reads', copied.run(m.var_get, plain), ('unset',))
copied.run(m.var_set, plain, 'inside')
eq('a value set inside one', copied.run(m.var_get, plain), ('value', 'inside'))
eq('and the context it was set from', m.var_get(plain), ('unset',))

# Every one of these opens by checking what it was handed.
for name, call in [
        ('get', lambda: m.var_get(object())),
        ('set', lambda: m.var_set(object(), 1)),
        ('reset', lambda: m.var_reset(object(), token))]:
    try:
        call()
    except TypeError:
        pass
    else:
        raise AssertionError('%s took something that is not a variable' % name)

print('cpyext-contextvars-ok')
"#;

const REPORT_SCRIPT: &str = r#"
import sys

import cpyext_context as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


seen = []


def recorder(kind, value, traceback):
    seen.append((kind, str(value), traceback is None))


for name in ('last_exc', 'last_type', 'last_value', 'last_traceback'):
    if hasattr(sys, name):
        delattr(sys, name)

sys.excepthook = recorder
try:
    eq('reporting one', m.report('printex', 'reported', 1), None)
finally:
    sys.excepthook = sys.__excepthook__

eq('what the hook was handed', len(seen), 1)
kind, text, no_traceback = seen[0]
eq('the class', kind is ValueError, True)
eq('the message', text, 'reported')
# Raised from C, so nothing unwound and there is no traceback to hand over.
eq('the traceback', no_traceback, True)

# The four names a post-mortem reads the exception back from.
eq('sys.last_exc', type(sys.last_exc) is ValueError, True)
eq('and its message', str(sys.last_exc), 'reported')
eq('sys.last_type', sys.last_type is ValueError, True)
eq('sys.last_value', sys.last_value is sys.last_exc, True)
eq('sys.last_traceback', sys.last_traceback is None, True)

# The indicator is cleared by the report, so the next call starts clean.
eq('nothing is left pending', m.report('printex', 'again', 0), None)

# With the flag off the names are left as the previous report set them.
eq('sys.last_exc is untouched', str(sys.last_exc), 'reported')

# `PyErr_Print` is the flag-on spelling.
del sys.last_exc
seen.clear()
sys.excepthook = recorder
try:
    eq('reporting through the short spelling',
       m.report('print', 'short', 0), None)
finally:
    sys.excepthook = sys.__excepthook__
eq('the hook saw it', [(k is ValueError, t) for k, t, _ in seen],
   [(True, 'short')])
eq('and the flag was on', str(sys.last_exc), 'short')

# A hook that raises is reported as unraisable rather than left pending, so
# the exception it raised does not surface at the next C call.
def raises(kind, value, traceback):
    raise RuntimeError('the hook failed')


sys.excepthook = raises
try:
    m.report('printex', 'swallowed', 0)
finally:
    sys.excepthook = sys.__excepthook__

# Reporting with nothing to report is the caller's mistake.
eq('reporting nothing', m.report_nothing(), 'SystemError')

print('cpyext-report-ok')
"#;

#[test]
fn the_namespace_a_name_falls_back_to() {
    let fixtures = Fixtures::new("cpyext-builtins");
    fixtures.compile("cpyext_context");
    fixtures.expect_ok(BUILTINS_SCRIPT, &[], "cpyext-builtins-ok");
}

#[test]
fn the_context_variables_an_extension_reaches() {
    let fixtures = Fixtures::new("cpyext-contextvars");
    fixtures.compile("cpyext_context");
    fixtures.expect_ok(CONTEXTVARS_SCRIPT, &[], "cpyext-contextvars-ok");
}

#[test]
fn reporting_a_failure_the_caller_cannot_hand_back() {
    let fixtures = Fixtures::new("cpyext-report");
    fixtures.compile("cpyext_context");
    fixtures.expect_ok(REPORT_SCRIPT, &[], "cpyext-report-ok");
}
