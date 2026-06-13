# A non-default `__getattribute__` must suppress LOAD_METHOD self-binding.
# The override returns the raw function object (`type(self).__dict__[name]`),
# never a bound method, so `c.f(...)` must call `f(...)` with exactly the
# supplied positionals — no implicit self.  `compute_load_method_bound`
# (shared by the interpreter `load_method` and the blackhole
# `bh_load_method_self_fn`) used to infer the binding from MRO shape alone and
# wrongly prepend self, inflating `len(args)` by one.
#
# Straight-line so it stays in the interpreter: a hot loop calling through a
# custom `__getattribute__` trips a separate, pre-existing JIT crash.
class C:
    def f(*args):
        return len(args)

    def __getattribute__(self, name):
        return type(self).__dict__[name]


c = C()
print(c.f())
print(c.f(1))
print(c.f(1, 2))
