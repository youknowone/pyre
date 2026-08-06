# pyre-check: max-pypy-ratio=19
# Hot-loop str/unicode subscript and length over every string kind: ASCII /
# latin1 (1-byte code units), BMP (2-byte), and non-BMP astral (4-byte). The
# subscripts emit residual STRGETITEM/UNICODEGETITEM and the lengths STRLEN/
# UNICODELEN, exercising the backend's descr-driven item/length reads at each
# item size (item_size 1/2/4, with the STR null-terminator base adjustment).
# A rolling ordinal checksum makes a wrong offset or width a visibly wrong
# number rather than a silent pass. Deterministic; output asserted cpython==pypy.


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


def main():
    ascii_s = "The quick brown fox jumps over the lazy dog 0123456789!?"
    latin1_s = "café déjà vu naïve résumé — ¡Hola! ½¾ ©®µ"
    bmp_s = "αβγδεζ ελληνικά Ω ДЖЕМ кириллица 日本語 テスト"
    astral_s = "𝕒𝕓𝕔𝕕 𝟙𝟚𝟛𝟜 😀🎉🚀 𐀀𐀁 mixed astral"

    n = 60000
    print("ascii", hot_checksum(n, ascii_s))
    print("latin1", hot_checksum(n, latin1_s))
    print("bmp", hot_checksum(n, bmp_s))
    print("astral", hot_checksum(n, astral_s))
    print("len", hot_len(n, ascii_s), hot_len(n, bmp_s), hot_len(n, astral_s))


main()
