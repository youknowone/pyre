# pyre-check: gate=1
# `typedef.py` gives PyCode a `__weakref__` descriptor, so a caller can attach
# per-code state that expires with the code object.
import gc
import weakref


def target(a, b=2):
    return a + b


code = target.__code__
reference = weakref.ref(code)
assert reference() is code

# A weak-keyed mapping over code objects is the shape callers actually use.
table = weakref.WeakKeyDictionary()
table[code] = 'protected'
assert table[code] == 'protected'

# A compiled-on-the-fly code object has no enclosing `co_consts` holding it,
# so dropping the last reference must clear the weak reference and fire the
# callback.  A nested function's code would stay alive through its parent.
dropped = []
transient = compile('0', '<weakref probe>', 'eval')
watcher = weakref.ref(transient, lambda ref: dropped.append(True))
assert watcher() is transient
del transient
gc.collect()
assert watcher() is None
assert dropped == [True]
