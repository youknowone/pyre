# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=name_bound_loop,property_loop,class_receiver_loop,name_bound_class_receiver_loop,explicit_class_receiver_loop,method_descriptor_loop
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

    def property_loop(self, n):
        total = 0
        for _ in range(n):
            total += super().prop
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
    checks = (
        ("name-bound descriptors + cellvar self", child.name_bound_loop(N), 19 * N),
        ("fused property", child.property_loop(N), 7 * N),
        ("class receiver", Child.class_receiver_loop(N), 12 * N),
        ("name-bound class receiver", Child.name_bound_class_receiver_loop(N), 3 * N),
        ("explicit class receiver", Child.explicit_class_receiver_loop(N), 3 * N),
        ("method descriptor", ListChild().method_descriptor_loop(N), 5 * N),
    )
    for label, got, want in checks:
        if got != want:
            print("FAIL", label, got, want)
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
