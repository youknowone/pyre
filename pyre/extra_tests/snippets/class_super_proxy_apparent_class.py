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
