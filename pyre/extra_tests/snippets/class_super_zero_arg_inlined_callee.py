# pyre-check: gate=1
# Zero-argument `super()` in a method the hot loop INLINES, rather than in the
# method that owns the loop.  Both spellings reach `LOAD_SUPER_ATTR`, but only
# this one runs the fold's sub-walk with an inlined callee on the framestack,
# and that is the configuration `zero_arg_super_attr.py` (loop inside the
# super-bearing method) and `super_attr_inlined_callee.py` (the two-argument
# form) between them leave uncovered.
#
# The loop counts here are the whole point of the file.  `DEFAULT_THRESHOLD` is
# 1039, so a loop that runs a few hundred times proves only that the
# interpreter is right -- the JIT never sees it.  Each count below is above
# that, so the traces this pins are actually compiled.


class Base:
    def opts(self, plugin, name):
        return (plugin, name)


class Derived(Base):
    def opts(self, plugin, name):
        return super().opts(plugin, name)


# The callee is reached through a second object's method, so the super-bearing
# frame is neither the portal nor the loop's own frame.
class Manager:
    def register(self, obj, plugin, name):
        return obj.opts(plugin, name)


manager, derived, total = Manager(), Derived(), 0
for i in range(2000):
    total += len(manager.register(derived, i, "n"))
assert total == 4000, total

# The same shape without the extra object: `opts` is still an inlined callee of
# the module-level loop.
direct_total = 0
for i in range(2000):
    direct_total += len(derived.opts(i, "n"))
assert direct_total == 4000, direct_total

# The pluggy/pytest shape: the loop lives in a sibling method of the same
# object, so the super-bearing callee is inlined into a method frame.
class _PluginBase:
    def parse_hookimpl_opts(self, value):
        return value + 1


class _PluginManager(_PluginBase):
    def parse_hookimpl_opts(self, value):
        return super().parse_hookimpl_opts(value) + 1

    def register(self):
        total = 0
        for value in range(2000):
            total += self.parse_hookimpl_opts(value)
        return total


assert _PluginManager().register() == sum(range(2000)) + 4000

# A property getter reached through another object, and a classmethod likewise:
# both put the `super()` frame one level below the frame running the loop.
class PropBase:
    @property
    def val(self):
        return 1


class PropDerived(PropBase):
    @property
    def val(self):
        return super().val + 1


class Reader:
    def read(self, obj):
        return obj.val


reader, prop_obj, prop_total = Reader(), PropDerived(), 0
for _ in range(2000):
    prop_total += reader.read(prop_obj)
assert prop_total == 4000, prop_total


class CmBase:
    @classmethod
    def make(cls, x):
        return (cls.__name__, x)


class CmDerived(CmBase):
    @classmethod
    def make(cls, x):
        return super().make(x)


class Builder:
    def build(self, cls, x):
        return cls.make(x)


builder, cm_total = Builder(), 0
for i in range(2000):
    cm_total += len(builder.build(CmDerived, i))
assert cm_total == 4000, cm_total

print("ok")
