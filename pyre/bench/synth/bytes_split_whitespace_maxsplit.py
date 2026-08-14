# pyre-check: max-pypy-ratio=65
# bytes/bytearray split/rsplit on whitespace (sep=None) with a positive
# maxsplit keeps the surrounding whitespace of the final remainder field,
# matching str. Output verified against CPython/PyPy.
# Trip count kept clear of the major-collection threshold check.py pins: at
# the previous 470000 this loop crossed it, and the eval-breaker bailout that
# follows re-enters through a bridge whose guard can then fail once more,
# which moves guard_failures for reasons outside this fixture. Crossing
# resumes around 0.45x of the old count; the gated counters are unchanged.
N = 117500


def main():
    hits = 0
    for _ in range(N):
        ok = (
            b"  a  b  ".split(None, 1) == [b"a", b"b  "]
            and b"  a  b  c  ".rsplit(None, 1) == [b"  a  b", b"c"]
            and bytearray(b"x  y  z  ").split(None, 1)
            == [bytearray(b"x"), bytearray(b"y  z  ")]
            and b"a b c".split(None, 1) == [b"a", b"b c"]
        )
        if ok:
            hits += 1
    print(hits)


main()
