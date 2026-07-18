# bytes/bytearray split/rsplit on whitespace (sep=None) with a positive
# maxsplit keeps the surrounding whitespace of the final remainder field,
# matching str. Output verified against CPython/PyPy.
N = 20000


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
