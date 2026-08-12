# pyre-check: no-cpython
import gc


class RuntimeOwner:
    pass


owner = RuntimeOwner()
proxy = super(RuntimeOwner, owner)
ordinary = repr(proxy)
direct = super.__repr__(proxy)

assert ordinary == direct == "<super: <class 'RuntimeOwner'>, <RuntimeOwner object>>"
assert any(obj is ordinary for obj in gc.get_objects())
assert any(obj is direct for obj in gc.get_objects())

print("super repr results are collectable")
