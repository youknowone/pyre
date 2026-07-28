import gc
from types import ModuleType


# PyPy allocates these builtin-layout subtypes through space.allocate_instance,
# so user-defined finalizers run once the instances become unreachable.
finalized = []


class FinalizingStaticmethod(staticmethod):
    def __del__(self):
        finalized.append("staticmethod")


class FinalizingClassmethod(classmethod):
    def __del__(self):
        finalized.append("classmethod")


class FinalizingProperty(property):
    def __del__(self):
        finalized.append("property")


class FinalizingModule(ModuleType):
    def __del__(self):
        finalized.append("module")


objects = [
    FinalizingStaticmethod(lambda: None),
    FinalizingClassmethod(lambda: None),
    FinalizingProperty(lambda self: None),
    FinalizingModule("finalizing_module"),
]
del objects
gc.collect()

assert sorted(finalized) == ["classmethod", "module", "property", "staticmethod"]

print("OK")
