# pyre-check: max-pypy-ratio=20
# Measured macOS dynasm 6.9x, macOS cranelift 7.7x, ubuntu cranelift 9.7x, with `classify` inlined into the loop.
# What is left is the rejection itself: the type call builds the
# UnicodeEncodeError eagerly on every iteration.
# The hot loop is the rejection alone, and N keeps pypy's execution time
# well clear of its startup. With the two accepted names in the loop every
# iteration left two dead types behind, and the ratio followed the length
# of `object`'s subclass list rather than the rejection.
# 67f223fe51d walks 3-arg type() through descr__new__ and classifies
# except-as-return, so the rejection compiles two loops and one bridge:
# t2/1:200 (plain eagerness, not a TY_REF split) + t1/37:1. Main and HEAD
# agree on every backend (gf=201 br=1 lp=2). The hot path raises
# UnicodeEncodeError and does not allocate a heap type.
N = 1000000
# Accepted names are checked a bounded number of times.
M = 100

def classify(name):
    try:
        return type(name, (), {}).__name__
    except UnicodeEncodeError as e:
        # codec spelling differs across runtimes, so encode only the
        # runtime-agnostic attributes into the checksum.
        return ("E", e.start, e.end, e.reason, e.object)


def main():
    lone = '\udcff'
    emb = 'a' + chr(0xd800) + 'b'
    astral = 'A' + chr(0x1f600)   # 4-byte, NOT a surrogate

    acc = 0
    i = 0
    while i < N:
        # hot path: rejection must not panic and must be repeatable
        r = classify(lone)
        if r[0] == "E" and r[1] == 0 and r[2] == 1 and r[3] == 'surrogates not allowed':
            acc = acc + 1
        i = i + 1

    # valid + astral names construct fine
    j = 0
    while j < M:
        if classify('Ok') == 'Ok':
            acc = acc + 1
        if classify(astral) == astral:
            acc = acc + 1
        j = j + 1

    # embedded surrogate reports the inner code-point position
    re = classify(emb)
    acc = acc + (re[1] if re[0] == "E" else -1)   # +1
    acc = acc + (re[2] if re[0] == "E" else -1)   # +2
    # object round-trips the original surrogate string
    acc = acc + (1 if classify(lone)[4] == lone else 0)

    print(acc)


main()
