//! The interpreter state a call runs inside: the namespace a name falls back
//! to, and the context variables.

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
