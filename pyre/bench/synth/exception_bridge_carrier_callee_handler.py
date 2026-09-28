# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# An exception-guard bridge whose raising call sits inside an inlined callee
# must resume that callee at its own `except` handler.
#
# `same` is inlined into `main`'s loop, and the `seq1.index(*args)` residual
# call inside it raises `ValueError` on a miss.  The bridge compiled for that
# guard resumes `same` through the multi-frame carrier.  When the carrier
# resumed `same` at the call's no-exception continuation instead, it took the
# `else:` branch with no `expected` bound, and `seq2.index` then raised out of
# `main`.  `Ctx.__exit__` clears the frames of the traceback it swallows, as
# `unittest`'s `assertRaises` does, which is what routes the guard failure
# into the carrier.


class Ctx:
    def __enter__(self):
        return self

    def __exit__(self, t, e, tb):
        tb = tb.tb_next
        while tb is not None:
            tb.tb_frame.clear()
            tb = tb.tb_next
        return True


class S:
    def __init__(self, seq):
        self.seq = seq

    def __getitem__(self, index):
        return self.seq[index]

    def __len__(self):
        return len(self.seq)

    def index(self, value, start=0, stop=None):
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)
        i = start
        while stop is None or i < stop:
            try:
                v = self[i]
            except IndexError:
                break
            if v is value or v == value:
                return i
            i += 1
        raise ValueError


SEEN = {}


def same(seq1, seq2, args):
    try:
        expected = seq1.index(*args)
    except ValueError:
        with Ctx():
            seq2.index(*args)
        key = "handler"
    else:
        actual = seq2.index(*args)
        assert actual == expected
        key = "else"
    SEEN[key] = SEEN.get(key, 0) + 1


def main():
    for ty in list, str:
        nat = ty("abracadabra")
        ss = S(nat)
        for letter in sorted(set(nat) | {"z"}):
            same(nat, ss, (letter,))
            for start in range(-3, 14):
                same(nat, ss, (letter, start))
                for stop in range(-3, 14):
                    same(nat, ss, (letter, start, stop))


main()
assert SEEN == {"else": 938, "handler": 2746}, SEEN
print("PASS")
