# pyre-check: max-pypy-ratio=19
# pyre-check: spec-folds=builtin_len
# Hot-loop str/unicode subscript and length over every string kind: ASCII /
# latin1 (1-byte code units), BMP (2-byte), and non-BMP astral (4-byte). The
# subscripts emit residual STRGETITEM/UNICODEGETITEM and the lengths STRLEN/
# UNICODELEN, exercising the backend's descr-driven item/length reads at each
# item size (item_size 1/2/4, with the STR null-terminator base adjustment).
# A rolling ordinal checksum makes a wrong offset or width a visibly wrong
# number rather than a silent pass. Deterministic; output asserted cpython==pypy.
#
# `hot_len` also runs over `bytes` and `bytearray`. The `bytes` arm reads
# `W_BytesObject.len` — the precomputed count standing in for the `strlen`
# `bytesobject.py` takes off `_value` — and the `bytearray` arm reads
# `W_BytearrayObject.length`, which is what `bytearrayobject.py`'s `_len`
# reaches through `self._data` as `rlist.py`'s `("length", Signed)`. Without
# those arms the call stays an opaque forcing residual and the leg measures
# ~100x the str one, which the ceiling below is set to catch.
#
# `hot_mutating_len` is a correctness leg, not a speed one. `bytearray`'s
# length is MUTABLE, so the compiled loop re-reads the field instead of
# hoisting it, and the field only stays right because every length-changing
# mutator republishes it through `w_bytearray_sync_alloc`. Appending and
# deleting a prefix inside the loop walks both the push path and the one that
# moves `logical_offset`; a length that went stale anywhere shows up as a wrong
# checksum here rather than as silently wrong compiled code.


def hot_checksum(n, s):
    length = len(s)
    acc = 0
    for i in range(n):
        acc = (acc * 131 + ord(s[i % length]) + length) & 0xFFFFFFFFFFFF
    return acc


def hot_len(n, s):
    acc = 0
    for i in range(n):
        acc += len(s)
    return acc


def hot_mutating_len(n, ba):
    acc = 0
    for i in range(n):
        ba.append(i & 0xFF)
        acc = (acc + len(ba)) & 0xFFFFFFFFFFFF
        if len(ba) > 64:
            del ba[0:32]
    return acc


def main():
    ascii_s = "The quick brown fox jumps over the lazy dog 0123456789!?"
    latin1_s = "café déjà vu naïve résumé — ¡Hola! ½¾ ©®µ"
    bmp_s = "αβγδεζ ελληνικά Ω ДЖЕМ кириллица 日本語 テスト"
    astral_s = "𝕒𝕓𝕔𝕕 𝟙𝟚𝟛𝟜 😀🎉🚀 𐀀𐀁 mixed astral"

    ascii_b = b"The quick brown fox jumps over the lazy dog 0123456789!?"
    short_b = b"abc"

    n = 60000
    print("ascii", hot_checksum(n, ascii_s))
    print("latin1", hot_checksum(n, latin1_s))
    print("bmp", hot_checksum(n, bmp_s))
    print("astral", hot_checksum(n, astral_s))
    print("len", hot_len(n, ascii_s), hot_len(n, bmp_s), hot_len(n, astral_s))
    # The bytes legs run longer than the str ones. Folded, a `len(bytes)` is a
    # single immutable field read, so a leg the size of the str ones would not
    # clear the measurement floor; unfolded it is ~100x that. Sized so that
    # losing the bytes arm ALONE carries the fixture past the ceiling above --
    # measured 2.1x pypy with the arm and 22x without it at a quarter of this
    # size. The ceiling itself is left where the str legs put it, because those
    # legs are what set it on the slowest runner.
    bn = 12000000
    print("blen", hot_len(bn, ascii_b), hot_len(bn, short_b))
    ban = 12000000
    print("balen", hot_len(ban, bytearray(ascii_b)), hot_len(ban, bytearray(short_b)))
    print("bamut", hot_mutating_len(400000, bytearray(short_b)))


main()
