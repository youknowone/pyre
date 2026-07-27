# str subscripts resolve a code point index, not a byte offset, for every
# payload width, and a slice keeps that indexing under negative and skipping
# steps. bytes/bytearray iterate their live buffer as ordinals, so a bytearray
# resized mid-loop is observed by the cursor rather than frozen at iter() time.
# The warm loops run the ASCII subscript and the bytes cursor hot first so the
# traced paths are the ones under test. Output verified against CPython/PyPy.
ASCII = "abcdefghij"
WIDE = "\xe9一\U0001f600ÿあ"
MIXED = "a一b\U0001f600c"


def warm_index(n):
    acc = 0
    for i in range(n):
        acc += ord(ASCII[i % 10])
    return acc


def warm_bytes(n):
    payload = bytes(range(64))
    acc = 0
    for _ in range(n):
        for byte in payload:
            acc += byte
    return acc


def show(label, fn):
    try:
        print(label, "->", ascii(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, ascii(str(e)))


def main():
    print("warm_index", warm_index(20000))
    print("warm_bytes", warm_bytes(400))

    for name, s in (("ascii", ASCII), ("wide", WIDE), ("mixed", MIXED)):
        print(name, "len", len(s))
        for i in range(-len(s), len(s)):
            show(f"{name}[{i}]", lambda s=s, i=i: s[i])
        show(f"{name}[oob]", lambda s=s: s[len(s)])
        show(f"{name}[-oob]", lambda s=s: s[-len(s) - 1])
        for sl in ((None, None, 2), (1, 4, None), (None, None, -1), (4, 0, -2), (-2, None, None)):
            show(f"{name}[{sl}]", lambda s=s, sl=sl: s[slice(*sl)])
        show(f"{name}.iter", lambda s=s: list(iter(s)))
        show(f"{name}.for", lambda s=s: [c for c in s])
        show(f"{name}.rev", lambda s=s: list(reversed(s)))

    show("bytes.iter", lambda: list(iter(b"abc")))
    show("bytes.for", lambda: [x for x in b"abc"])
    show("bytearray.iter", lambda: list(iter(bytearray(b"abc"))))
    show("bytes.empty", lambda: list(iter(b"")))

    def grow():
        buf = bytearray(b"abc")
        seen = []
        for x in buf:
            seen.append(x)
            if len(seen) == 1:
                buf.append(100)
        return seen

    def shrink():
        buf = bytearray(b"abcdef")
        seen = []
        for x in buf:
            seen.append(x)
            if len(seen) == 1:
                del buf[2:]
        return seen

    show("bytearray.grow", grow)
    show("bytearray.shrink", shrink)


main()
