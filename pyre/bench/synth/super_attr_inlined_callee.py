# pyre-check: spec-folds=two_arg_super_call,load_attr_on_super,super_attr_unwrap
# `super(C, self).m(x)` in a callee the FOR_ITER-driven trace inlines.
# `load_super_attr.py` keeps its loop inside the super-bearing method, so the
# callee replay scan never sees the LOAD_SUPER_ATTR residuals; this shape does.
#
# The two-argument oparg form's three folds have to fire from INSIDE an inlined
# callee, where `two_arg_super_call` additionally asks that the caller be paused
# on the framestack with its own parent frame -- which it is here, because the
# call this callee was inlined at is an ordinary CALL.
#
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 20000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def val(self, x):
        return super(Child, self).val(x)


def run(o, n):
    acc = 0
    for _ in range(n):
        acc = o.val(acc)
    return acc


print(run(Child(), N))
