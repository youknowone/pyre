# A list subscript, assignment, and deletion coerce a non-int key through
# `__index__` (`getindex_w`); an index too large for a machine word raises
# IndexError naming the key's real type ("cannot fit '<type>' ..."), and an
# out-of-range index raises "list index out of range".  A plain-int warmup
# loop exercises the working integer fast paths first.  Only cpython==pypy
# outputs are asserted (the "indices must be integers" TypeError text and the
# assignment out-of-range message diverge between the oracles).  Deterministic.
class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v


class Big:
    def __index__(self):
        return 10 ** 40


def warm(n):
    acc = 0
    data = [0, 1, 2, 3, 4]
    for i in range(n):
        j = i % 5
        data[j] = data[j] + 1
        acc += data[j]
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def read(i):
    return [10, 20, 30][i]


def setv(i):
    l = [0, 0, 0]
    l[i] = 9
    return l


def delv(i):
    l = [0, 1, 2]
    del l[i]
    return l


def main():
    print("warm", warm(15000))
    m("read_idx", lambda: read(Idx(1)))
    m("read_neg", lambda: read(Idx(-1)))
    m("read_oob", lambda: read(Idx(9)))
    m("read_big", lambda: read(Big()))
    m("set_idx", lambda: setv(Idx(1)))
    m("set_neg", lambda: setv(Idx(-1)))
    m("del_idx", lambda: delv(Idx(1)))
    m("del_neg", lambda: delv(Idx(-1)))
    m("del_big", lambda: delv(Big()))


main()
