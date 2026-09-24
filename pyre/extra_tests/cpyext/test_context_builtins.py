# cpyext-fixture: cpyext_context
# cpyext-expect: cpyext-builtins-ok

# The interpreter state a call runs inside: the namespace a name falls back
# to, the context variables, and reporting a failure the caller has no way to
# hand back.

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
