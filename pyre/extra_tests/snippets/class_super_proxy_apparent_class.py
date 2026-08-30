# pyre-check: gate=1
class Proxy:
    def __init__(self, obj):
        self.__obj = obj
    def __getattribute__(self, name):
        if name.startswith('_Proxy__'):
            return object.__getattribute__(self, name)
        return getattr(self.__obj, name)
class B:
    def f(self):
        return 'B.f'
class C(B):
    def f(self):
        return super(C, self).f() + '->C.f'
result = C.__dict__['f'](Proxy(C()))

assert result == 'B.f->C.f'

# The name binding splits the proxy from the attribute load, so the second half
# arrives as an ordinary `LOAD_ATTR` on a `W_Super` rather than as
# `LOAD_SUPER_ATTR`.  `W_Super.getattribute` walks the `w_objtype` stored at
# construction -- recomputing `type(w_self)` here would answer `Proxy` and find
# no `f` at all -- so the loop must keep answering `B.f` once it compiles.
class D(B):
    def f(self):
        su = super(D, self)
        return su.f() + '->D.f'

proxy = Proxy(D())
hot = [D.__dict__['f'](proxy) for _ in range(5000)]
assert hot == ['B.f->D.f'] * 5000, hot[:3]
