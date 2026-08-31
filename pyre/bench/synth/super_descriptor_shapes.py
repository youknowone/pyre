# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=name_bound_loop,slot_wrapper_loop,property_loop,class_receiver_loop,name_bound_class_receiver_loop,explicit_class_receiver_loop,method_descriptor_loop,apparent_fused_loop,apparent_property_loop,apparent_name_bound_loop,apparent_explicit_loop
# Correctness and compilation coverage for the descriptor shapes selected by
# `W_Super.getattribute`.  PyPy traces the unrolled MRO suffix walk and then the
# ordinary descriptor protocol; pyre's fused LOAD_SUPER_ATTR and name-bound
# proxy paths must preserve the same binding while eliminating their opaque
# residuals.
N = 20000


class Base:
    k = 3

    @staticmethod
    def sm(x):
        return x + 1

    @classmethod
    def cm(cls):
        return cls.bias

    @property
    def prop(self):
        return self.bias

    @property
    def fixed(self):
        return 5


class Child(Base):
    bias = 7

    def name_bound_loop(self, n):
        # Capturing `self` makes fast-local slot zero a Cell.  The zero-argument
        # super fold must read Cell.contents rather than treating the Cell as
        # the receiver.
        def captured():
            return self

        total = 0
        for _ in range(n):
            su = super()
            total += su.k + su.sm(1) + su.cm() + su.prop
        if captured() is not self:
            return -1
        return total

    def slot_wrapper_loop(self, n):
        # `object.__init__` is a slot wrapper, so `get` takes the arm that
        # binds through `method_wrapper_new`: the same `Method` payload the
        # method-descriptor arm builds, restamped `method-wrapper` instead of
        # `builtin_function_or_method`.
        total = 0
        for _ in range(n):
            super().__init__()
            total += 1
        return total

    def property_loop(self, n):
        total = 0
        for _ in range(n):
            total += super().prop
        return total

    def apparent_fused_loop(self, n):
        total = 0
        for _ in range(n):
            total = super().sm(total)
        return total

    def apparent_property_loop(self, n):
        total = 0
        for _ in range(n):
            total += super().fixed
        return total

    def apparent_name_bound_loop(self, n):
        total = 0
        for _ in range(n):
            proxy = super()
            total += proxy.k
        return total

    def apparent_explicit_loop(self, n):
        total = 0
        for _ in range(n):
            proxy = super(Child, self)
            total += proxy.k
        return total

    @classmethod
    def class_receiver_loop(cls, n):
        total = 0
        for _ in range(n):
            # `super().__getattribute__` binds these with descr_obj=None:
            # constants/functions stay unbound, while classmethod still binds
            # to the walked receiver class.
            total += super().k + super().sm(1) + super().cm()
        return total

    @classmethod
    def name_bound_class_receiver_loop(cls, n):
        total = 0
        for _ in range(n):
            proxy = super()
            total += proxy.k
        return total

    @classmethod
    def explicit_class_receiver_loop(cls, n):
        total = 0
        for _ in range(n):
            proxy = super(Child, cls)
            total += proxy.k
        return total


class ListChild(list):
    def method_descriptor_loop(self, n):
        total = 0
        for _ in range(n):
            super().append(5)
            total += super().pop()
        return total


class ApparentChild:
    """Its installed type is unrelated; only the traced attribute says Child."""

    __class__ = Child


class Sneaky(Base):
    # A slot wrapper this class does not own.  Binding it to a `Sneaky` fails
    # the owner subtype test that runs before the binding, so the fold has to
    # decline and leave the raise to the interpreter.
    __init__ = list.__init__


class SneakyChild(Sneaky):
    def reject_loop(self, n):
        rejected = 0
        for _ in range(n):
            try:
                super().__init__()
            except TypeError:
                rejected += 1
        return rejected


class Override(super):
    def __getattribute__(self, name):
        if name == "k":
            return 99
        if name == "\udcff":
            return 88
        return super().__getattribute__(name)


class Fallback(super):
    def __getattr__(self, name):
        if name == "missing":
            return 77
        if name == "\udcfe":
            return 66
        raise AttributeError(name)


def main():
    child = Child()
    apparent = ApparentChild()
    checks = (
        ("name-bound descriptors + cellvar self", child.name_bound_loop(N), 19 * N),
        ("slot wrapper", child.slot_wrapper_loop(N), N),
        ("fused property", child.property_loop(N), 7 * N),
        ("class receiver", Child.class_receiver_loop(N), 12 * N),
        ("name-bound class receiver", Child.name_bound_class_receiver_loop(N), 3 * N),
        ("explicit class receiver", Child.explicit_class_receiver_loop(N), 3 * N),
        ("method descriptor", ListChild().method_descriptor_loop(N), 5 * N),
        ("apparent class fused", Child.apparent_fused_loop(apparent, N), N),
        (
            "apparent class property",
            Child.apparent_property_loop(apparent, N),
            5 * N,
        ),
        (
            "apparent class name-bound",
            Child.apparent_name_bound_loop(apparent, N),
            3 * N,
        ),
        (
            "apparent class explicit",
            Child.apparent_explicit_loop(apparent, N),
            3 * N,
        ),
        ("rejected slot wrapper", object.__new__(SneakyChild).reject_loop(N), N),
    )
    for label, got, want in checks:
        if got != want:
            print("FAIL", label, got, want)
            return 1

    bound_init = super(Child, child).__init__
    # pypy3 is 3.11 and answers `method` here; 3.14 restamps the binding.
    if type(bound_init).__name__ != "method-wrapper":
        print("FAIL slot wrapper public class", type(bound_init).__name__)
        return 1
    if bound_init.__self__ is not child:
        print("FAIL slot wrapper receiver")
        return 1

    proxy = Override(Child, child)
    if proxy.k != 99:
        print("FAIL super subclass override", proxy.k)
        return 1
    if getattr(proxy, "\udcff") != 88:
        print("FAIL super subclass surrogate override")
        return 1
    if proxy.__thisclass__ is not Child:
        print("FAIL inherited super getattribute")
        return 1
    if super.__getattribute__(proxy, "k") != 3:
        print("FAIL explicit builtin super getattribute")
        return 1
    fallback = Fallback(Child, child)
    if fallback.missing != 77:
        print("FAIL super subclass getattr fallback")
        return 1
    if getattr(fallback, "\udcfe") != 66:
        print("FAIL super subclass surrogate getattr fallback")
        return 1
    print("PASS", sum(got for _, got, _ in checks))
    return 0


raise SystemExit(main())
