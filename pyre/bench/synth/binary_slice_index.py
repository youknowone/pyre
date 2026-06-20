# BINARY_SLICE evaluates each non-None bound through __index__
# (eval_slice_index) instead of reading the raw int field, so a custom
# __index__, a bool, and a non-int __index__ result behave like the
# slice(start, stop) + __getitem__ fallback.  Only the exception type is
# printed so the line matches across CPython/PyPy/Pyre.
#
# (A plain float bound is a separate, pre-existing interpreter gap — getindex_w
# reads a float's bits as an integer instead of raising — so it is not
# exercised here.)
N = 50000


class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v


class BadIdx:
    def __index__(self):
        return "not-an-int"


def show(label, fn):
    try:
        print(label, fn())
    except Exception as e:
        print(label, type(e).__name__)


def main():
    seq = "abcdefghij"
    lst = list(range(10))
    tup = tuple(range(10))

    # Hot loop: the JIT-compiled BINARY_SLICE residual must run __index__ on
    # the instance bounds every iteration.  A fast path that read the raw int
    # field of the bound object would mis-slice or crash on the instance.
    acc = 0
    i = 0
    while i < N:
        a = Idx(i % 5)
        b = Idx(i % 5 + 3)
        acc = acc + len(seq[a:b]) + len(lst[a:b]) + len(tup[a:b])
        i = i + 1
    print("acc", acc)

    # __index__ bounds resolve to the same slice as plain ints.
    show("str_idx", lambda: seq[Idx(2):Idx(6)])
    show("list_idx", lambda: sum(lst[Idx(2):Idx(6)]))
    show("tuple_idx", lambda: sum(tup[Idx(2):Idx(6)]))

    # bool is an int subtype; True/False are valid bounds.
    show("bool_bound", lambda: seq[True:4])

    # An __index__ that returns a non-int -> TypeError.
    show("badidx_bound", lambda: seq[BadIdx():4])

    # None bounds still default to 0 / len.
    show("none_start", lambda: seq[:4])
    show("none_stop", lambda: sum(lst[6:]))

    # Code-point-boundary slicing over a lone surrogate, with __index__ bounds.
    surr = "a\ud800b"
    show("surrogate_len", lambda: len(surr[Idx(1):Idx(3)]))
    show("surrogate_ord", lambda: ord(surr[Idx(1):Idx(2)]))


main()
