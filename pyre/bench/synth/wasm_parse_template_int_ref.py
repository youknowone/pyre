# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main,root:parse_template
# A wasm-only JIT miscompile returned the integer group index from
# `re._parser.parse_template` as a bogus object:
#
#     ['', <object object at 0x9>, '']
#
# instead of `['', 1, '']`.  The trigger needs a generic-iterator FOR_ITER
# around the call; a `range()` loop does not reproduce it.  The native
# backends answered correctly.
#
# This is a selfcheck rather than an oracle comparison because the private
# parser API changed between the available PyPy 3.11 oracle and pyre's pinned
# 3.14 stdlib.  PyPy returns `([(0, 1)], [None])`, while the pinned parser
# returns the interleaved list checked here.  The observable being guarded is
# internal consistency within that pinned implementation: a group index built
# as an int must remain an int when a compiled caller receives it.
#
# `main` is the generic FOR_ITER loop that made the defect visible, while
# `root:parse_template` proves the parser path also reached compiled code.  A
# cold or wholly residual run therefore cannot satisfy this fixture.
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
    return bad


bad = main()
if bad == 0:
    print("PASS wasm parse-template int reference")
else:
    print(f"FAIL parse-template returned {bad} non-int group references")
    raise SystemExit(1)
