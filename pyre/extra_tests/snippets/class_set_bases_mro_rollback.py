# pyre-check: gate=1
class A:
    pass
class B:
    pass
class C(A, B):
    pass
class D(A, B):
    pass
class E(C, D):
    pass
old_bases = C.__bases__
old_mro = C.__mro__
rejected = False
try:
    C.__bases__ = (B, A)
except TypeError:
    rejected = True
restored = C.__bases__ == old_bases and C.__mro__ == old_mro

assert rejected
assert restored
