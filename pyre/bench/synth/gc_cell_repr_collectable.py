# pyre-check: no-cpython
import gc


def make_cell(value):
    def inner():
        return value

    return inner.__closure__[0]


filled = make_cell(42)
empty = make_cell(42)
del empty.cell_contents

for cell in (filled, empty):
    ordinary = repr(cell)
    direct = type(cell).__repr__(cell)

    assert ordinary == direct
    assert any(obj is ordinary for obj in gc.get_objects())
    assert any(obj is direct for obj in gc.get_objects())

print("cell repr results are collectable")
