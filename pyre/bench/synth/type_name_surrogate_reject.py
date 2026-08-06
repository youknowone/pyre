# pyre-check: max-pypy-ratio=26
N = 10000


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
        # valid + astral names construct fine
        if classify('Ok') == 'Ok':
            acc = acc + 1
        if classify(astral) == astral:
            acc = acc + 1
        i = i + 1

    # embedded surrogate reports the inner code-point position
    re = classify(emb)
    acc = acc + (re[1] if re[0] == "E" else -1)   # +1
    acc = acc + (re[2] if re[0] == "E" else -1)   # +2
    # object round-trips the original surrogate string
    acc = acc + (1 if classify(lone)[4] == lone else 0)

    print(acc)


main()
