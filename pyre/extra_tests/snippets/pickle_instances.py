import pickle
class C:
    def __init__(self, x): self.x = x
    def __eq__(self, o): return type(self) is type(o) and self.x == o.x
class S:
    __slots__ = ("a",)
    def __init__(self, a): self.a = a
    def __eq__(self, o): return type(self) is type(o) and self.a == o.a
for proto in range(0, 6):
    assert pickle.loads(pickle.dumps(C(42), proto)) == C(42)
    assert pickle.loads(pickle.dumps(C([1,2]), proto)) == C([1,2])
    nested = C(C(C(1)))
    assert pickle.loads(pickle.dumps(nested, proto)).x.x.x == 1
# A __slots__ class without a custom __getstate__ can only be pickled at
# protocol >= 2 (copyreg._reduce_ex raises for protocols 0 and 1).
for proto in range(2, 6):
    assert pickle.loads(pickle.dumps(S(7), proto)) == S(7)
print("pickle_instances OK")
