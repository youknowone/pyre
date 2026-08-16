# pyre-check: gate=1
class B:
    pass
class Meta(type):
    def mro(cls):
        del Meta.mro
        return (B,)
rejected = False
try:
    class A(metaclass=Meta):
        pass
except TypeError:
    rejected = True

assert rejected
