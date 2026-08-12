# pyre-check: no-cpython
import gc


class RuntimeOwner:
    def runtime_method(self):
        pass


owner = RuntimeOwner()
method = owner.runtime_method
ordinary = repr(method)
direct = type(method).__repr__(method)

assert ordinary == direct
assert "RuntimeOwner.runtime_method" in ordinary
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("bound method repr results are collectable")
