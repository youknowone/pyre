# wasm-only JIT miscompile: `re._parser.parse_template` returns the int group
# index as a bogus object (`['', <object object at 0x9>, '']` instead of
# `['', 1, '']`).  Needs a GENERIC-iterator FOR_ITER around the call; a
# `range()` loop does not trigger it.  dynasm/cranelift print 0.
import re
import re._parser as rp

pair = re.compile(r"([a-z]+)(\d+)")


def main():
    items = [1, 2, 3]
    bad = 0
    i = 0
    while i < 1000:
        for m in items:
            t = rp.parse_template("\\1", pair)
            if not isinstance(t[1], int):
                bad = bad + 1
        i = i + 1
    print(bad)


main()
