# Changing an unboxed int or float slot to the other type creates a boxed map;
# the promoted-map guard must deopt before the old raw-storage read or write is
# reused for the new representation (mapdict.py:577-584, 600-619, 905-916).
class IntSlot:
    def __init__(self):
        self.x = 0


class FloatSlot:
    def __init__(self):
        self.x = 0.0


def run_int(n):
    obj = IntSlot()
    total = 0
    i = 0
    while i < n:
        obj.x = obj.x + 1
        if i == n // 2:
            obj.x = 1.5
        total += int(obj.x)
        i += 1
    return total, obj.x


def run_float(n):
    obj = FloatSlot()
    total = 0.0
    i = 0
    while i < n:
        obj.x = obj.x + 1.0
        if i == n // 2:
            obj.x = 7
        total += float(obj.x)
        i += 1
    return total, obj.x


print(run_int(100000))
print(run_float(100000))
