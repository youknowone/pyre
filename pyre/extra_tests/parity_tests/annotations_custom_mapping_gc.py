# CPython-suite gap: SETUP_ANNOTATIONS must root the newly-created dictionary
# while an arbitrary class namespace runs Python code during item assignment.
# parity-tests reason: exercise collection at the custom namespace boundary.

import gc


class Namespace:
    def __init__(self):
        self.storage = {}

    def __getitem__(self, key):
        gc.collect()
        return self.storage[key]

    def __setitem__(self, key, value):
        gc.collect()
        self.storage[key] = value

    def __contains__(self, key):
        raise AssertionError("SETUP_ANNOTATIONS must use item lookup")


class Meta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        return Namespace()

    def __new__(mcls, name, bases, namespace):
        return super().__new__(mcls, name, bases, namespace.storage)


source = """from __future__ import annotations
class C(metaclass=Meta):
    x: int
    y: str
"""
for _ in range(30):
    namespace = {"Meta": Meta, "__name__": "annotations_custom_mapping_gc"}
    exec(source, namespace)
    assert namespace["C"].__annotations__ == {"x": "int", "y": "str"}
print("OK")
