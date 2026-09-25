# cpyext-fixture: cpyext_context
# cpyext-expect: cpyext-contextvars-ok

# The interpreter state a call runs inside: the namespace a name falls back
# to, the context variables, and reporting a failure the caller has no way to
# hand back.

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
