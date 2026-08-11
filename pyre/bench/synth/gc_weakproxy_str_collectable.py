# pyre-check: no-cpython

import gc
import weakref


marker = "".join(["dynamic", " weak proxy"])


class Referent:
    def __str__(self):
        return marker


referent = Referent()
proxy = weakref.proxy(referent)
ordinary = str(proxy)
direct = type(proxy).__str__(proxy)

assert ordinary is marker
assert direct is marker
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("weak proxy str preserves managed identity")
