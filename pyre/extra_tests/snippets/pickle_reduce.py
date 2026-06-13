import copyreg, pickle
class C:
    def __init__(self): self.x = 1
r = C().__reduce_ex__(2)
assert r[0] is copyreg.__newobj__ and r[1] == (C,) and r[2] == {"x": 1} and r[3] is None and r[4] is None, r
r = (1, 2).__reduce_ex__(2)
assert r[0] is copyreg.__newobj__ and r[1] == (tuple, (1, 2)) and r[2] is None, r
r = C().__reduce_ex__(0)
assert r[0] is copyreg._reconstructor and r[1] == (C, object, None) and r[2] == {"x": 1}, r
assert C().__reduce__()[0] is copyreg._reconstructor
assert C().__getstate__() == {"x": 1}
class Empty: pass
assert Empty().__getstate__() is None and object().__getstate__() is None
class S:
    __slots__ = ("a", "b")
    def __init__(self): self.a = 1; self.b = 2
assert S().__reduce_ex__(2)[2] == (None, {"a": 1, "b": 2}), S().__reduce_ex__(2)
assert (1,2).__getnewargs__() == ((1,2),) and (5).__getnewargs__() == (5,) and "ab".__getnewargs__() == ("ab",)
try:
    (1, 2).__reduce_ex__(0); raise AssertionError("expected TypeError")
except TypeError as e:
    assert "cannot pickle" in str(e), e
print("pickle_reduce OK")
