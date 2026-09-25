# pyre-check: gate=1
# A warm self-recursive raise caught by a bare module-level `try` keeps one
# traceback node per frame: the module frame plus every `f` activation.


def f(n, boom):
    if n == 0:
        if boom:
            raise ValueError("bottom")
        return 0
    return f(n - 1, boom)


def tb_len(exc):
    tb = exc.__traceback__
    n = 0
    while tb is not None:
        n += 1
        tb = tb.tb_next
    return n


for boom in (False, True):
    for i in range(2000):
        try:
            f(8, boom)
        except ValueError:
            pass

try:
    f(8, True)
except ValueError as e:
    assert tb_len(e) == 10, tb_len(e)

for depth in (0, 1, 5, 30):
    try:
        f(depth, True)
    except ValueError as e:
        assert tb_len(e) == depth + 2, (depth, tb_len(e))
