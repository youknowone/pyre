# CPython-suite gap: no CPython test pins make_weakref_descr _lifeline_ on these interp types.
# parity-tests reason: typedef.py make_weakref_descr stores _lifeline_ on the instance.

"""Exact instances of make_weakref_descr types accept weakref.ref.

`typedef.py` `make_weakref_descr` publishes `__weakref__` and a `_lifeline_`
field. `_getusercls` does not mix `MapdictWeakrefSupport` for those TypeDefs,
so a subclass of a weakrefable base keeps the same field.
"""

import io
import types
import weakref
import _thread
from collections import deque


def alive(obj):
    return weakref.ref(obj)() is obj


def f():
    yield 1


gen = f()
mod = types.ModuleType("make_weakref_descr_lifeline")
lock = _thread.allocate_lock()
dq = deque()
bio = io.BytesIO()
local = _thread._local()


class SubLocal(_thread._local):
    pass


class SubMod(types.ModuleType):
    pass


sub_local = SubLocal()
sub_mod = SubMod("make_weakref_descr_lifeline_sub")

assert alive(f)
assert alive(gen)
assert alive(type)
assert alive(mod)
assert alive(lock)
assert alive(dq)
assert alive(bio)
assert alive(local)
assert alive(sub_local)
assert alive(sub_mod)
assert alive(set())
assert alive(frozenset())
print("OK")
