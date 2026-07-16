"""PyPy W_Super typedef and Python 3.14 super semantics."""


required = {
    "__doc__",
    "__new__",
    "__init__",
    "__repr__",
    "__getattribute__",
    "__get__",
    "__thisclass__",
    "__self__",
    "__self_class__",
}
assert required <= set(super.__dict__)
assert "super() -> same as" in super.__doc__


class A:
    value = "A"

    def method(self):
        return "A.method"

    @classmethod
    def class_method(cls):
        return cls.__name__


class B(A):
    value = "B"

    def method(self):
        return super().method() + ":B"

    @classmethod
    def class_method(cls):
        return super().class_method()


obj = B()
bound = super(B, obj)
assert repr(bound) == "<super: <class 'B'>, <B object>>"
assert bound.__thisclass__ is B
assert bound.__self__ is obj
assert bound.__self_class__ is B
assert bound.value == "A"
assert bound.method() == "A.method"
assert B().method() == "A.method:B"
assert B.class_method() == "B"

class_bound = super(B, B)
assert class_bound.__self__ is B
assert class_bound.__self_class__ is B
assert class_bound.class_method() == "B"

unbound = super(B)
assert repr(unbound) == "<super: <class 'B'>, NULL>"
assert unbound.__thisclass__ is B
assert unbound.__self__ is None
assert unbound.__self_class__ is None
assert unbound.__get__(obj, B).method() == "A.method"


class SuperSubclass(super):
    pass


sub = SuperSubclass(B, obj)
assert isinstance(sub, SuperSubclass)
assert sub.method() == "A.method"

for args in ((1,), (B, A()), (B, obj, obj)):
    try:
        super(*args)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid super arguments must fail")

print("OK")
