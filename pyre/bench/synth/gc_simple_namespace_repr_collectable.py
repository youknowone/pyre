# pyre-check: no-cpython

import gc
from types import SimpleNamespace


namespace = SimpleNamespace(alpha="value", count=3)
ordinary = repr(namespace)
direct = SimpleNamespace.__repr__(namespace)

assert ordinary == "namespace(alpha='value', count=3)"
assert direct == ordinary
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

recursive_result = None


class CaptureRecursiveRepr:
    def __repr__(self):
        global recursive_result
        recursive_result = repr(recursive_namespace)
        return "captured"


recursive_namespace = SimpleNamespace(value=CaptureRecursiveRepr())
assert repr(recursive_namespace) == "namespace(value=captured)"
assert recursive_result == "namespace(...)"
assert any(obj is recursive_result for obj in gc.get_objects())

print("SimpleNamespace repr results are collectable")
