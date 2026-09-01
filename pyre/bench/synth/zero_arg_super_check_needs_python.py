# pyre-check: max-pypy-ratio=150
# pyre-check: spec-folds=bare_super_virtual,load_attr_on_super
# The apparent-class twin of `zero_arg_name_bound_super_attr.py`: `type(self)`
# is unrelated to `Child`, while the ordinary class attribute
# `self.__class__` names it.  This reaches PyPy's third `_super_check` arm.
#
# PyPy traces that `space.getattr` like any other attribute read.  For this
# non-descriptor class attribute, mapdict promotes the receiver type/version
# and map, then returns the installed `Child` constant; `_super_check` promotes
# that type for its subtype test and stores it as `W_Super.w_objtype`.
# `bare_super_virtual` must reproduce exactly those guards.  In particular it
# must keep the receiver's real `w_class == Proxy` and must not replace that
# proof with `w_class == Child`, which is the normal `_super_check` arm and is
# false by construction here.
#
# Before that attribute trace was connected to super construction, the
# frame-explicit `bare_super_call` residual preserved correctness and kept the
# loop compilable, but paid `_super_check` once per iteration: dynasm was about
# 0.17s here against 0.06s for a real `Child` receiver.  The virtual fold now
# emits the proxy directly, and `load_attr_on_super` consumes its stored
# apparent type for `s.val` without re-deriving it from the receiver.
#
# Keep this as a benchmark rather than only a selfcheck: it holds both the
# zero-abort snapshot and the disappearance of the residual cost.  The broad
# ratio ceiling predates the virtual arm and remains a level target until a
# backend leg with an unclamped PyPy timing supplies a tighter stable number.
N = 1000000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def loop(self, n):
        acc = 0
        for _ in range(n):
            s = super()
            acc = s.val(acc)
        return acc


class Proxy:
    """`type(self)` is not a `Child`.  Only `self.__class__` says so."""

    __class__ = Child


print(Child.loop(Proxy(), N))
