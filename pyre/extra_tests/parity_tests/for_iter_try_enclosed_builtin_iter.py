# CPython-suite gap: the suite does not JIT-compile a hot builtin-iterator
# loop whose FOR_ITER sits inside a try range.
# parity-tests reason: FOR_ITER materializes its own catch, so the enclosing
# try's handler edge must not disturb the loop's exit shape — this holds for
# every iterator, not only user-defined ones.


def sum_range(n):
    total = 0
    try:
        for value in range(n):
            total += value
    except ValueError:
        return -1
    return total


for _ in range(12):
    assert sum_range(2000) == 1999 * 2000 // 2


def sum_list(values):
    total = 0
    try:
        for value in values:
            total += value
    except ValueError:
        return -1
    return total


data = list(range(2000))
for _ in range(12):
    assert sum_list(data) == 1999 * 2000 // 2


# Nested try ranges around the same loop, and a loop whose body itself raises
# into the enclosing handler.
def sum_nested(n, raise_at):
    total = 0
    try:
        try:
            for value in range(n):
                if value == raise_at:
                    raise ValueError("inner")
                total += value
        except TypeError:
            return -2
    except ValueError:
        return total
    return total


for _ in range(12):
    assert sum_nested(2000, -1) == 1999 * 2000 // 2

assert sum_nested(2000, 1500) == 1499 * 1500 // 2


# A generator iterator inside a try: exhaustion must still end the loop.
def gen(n):
    for i in range(n):
        yield i


def sum_gen(n):
    total = 0
    try:
        for value in gen(n):
            total += value
    except ValueError:
        return -1
    return total


for _ in range(12):
    assert sum_gen(1500) == 1499 * 1500 // 2

print("OK")
