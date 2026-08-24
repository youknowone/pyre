# pyre-check: spec-folds=subscr_str
# Hot-loop `s[i]` on an exact `str`. The subscript reaches the walker as the
# `BinaryOp` helper's `Subscr` tag, and `try_walker_specialize_subscr` only
# recognized dict, tuple, specialised-pair and list receivers -- a `str` fell
# through to the generic `CallMayForce`, which forces virtualizables and
# clears the heap cache across itself. Measured with the identical loop body
# over a `list` receiver, whose storage arm already folds, the str form cost
# an order of magnitude more per iteration than the one code point it boxes.
#
# NO throughput ceiling, deliberately. The fold removes the may-force
# residual and nothing else: 51.1 -> 45.5 ns/iter on dynasm, against pypy's
# 0.7. The remaining 45 is the boxed code point itself, which this loop
# discards immediately and pypy never allocates at all -- `len(s[0])` reads
# 44.3 and `ord(s[0])` 45.0, i.e. the cost does not depend on the consumer.
# Virtualizing that allocation is a different lever, so the two arms here sit
# at 73x and 64x pypy and no ratio between them would be a stable gate. The
# `spec-folds` line above is the gate; a ratio would only measure the
# allocation.
#
# The other legs are correctness legs for the shapes the fold must REFUSE,
# each written so a wrongly-admitted shape is a wrong value and not a silent
# pass:
#
#   * a `str` SUBCLASS may override `__getitem__`, which `baseobjspace::getitem`
#     honours; admitting one through the payload `ob_type` it shares would drop
#     the prefix its override adds.
#   * a NON-ASCII receiver indexes code points, not bytes; a fixed-stride read
#     would return a fragment of a multi-byte sequence.
#   * a NEGATIVE index counts from the end, and an out-of-range one raises
#     `IndexError` -- both belong to the interpreter, so the helper declines
#     them with `PY_NULL` and the non-null guard carries that back.
#   * a `bool` index shares `int`'s `intval` but carries its own type, and a
#     `__index__` object is not an int at all.
class Prefixed(str):
    def __getitem__(self, index):
        return "!" + str.__getitem__(self, index)


class Ix:
    def __index__(self):
        return 2


def hot_index(n, s):
    acc = 0
    for _ in range(n):
        for i in range(5):
            acc = (acc * 31 + ord(s[i])) & 0xFFFFFFFFFF
    return acc


def declined_shapes(n, plain, wide, sub):
    acc = 0
    for _ in range(n):
        acc = (acc * 31 + ord(wide[0]) + ord(wide[3])) & 0xFFFFFFFFFF
        acc = (acc * 31 + len(sub[0])) & 0xFFFFFFFFFF
        acc = (acc * 31 + ord(plain[-1]) + ord(plain[True])) & 0xFFFFFFFFFF
        acc = (acc * 31 + ord(plain[Ix()])) & 0xFFFFFFFFFF
        try:
            plain[99]
        except IndexError:
            acc = (acc * 31 + 7) & 0xFFFFFFFFFF
    return acc


PLAIN = "abcde"
WIDE = "aé中𝄞x"
SUB = Prefixed("abcde")

print(hot_index(300000, PLAIN))
print(declined_shapes(20000, PLAIN, WIDE, SUB))
print(WIDE[1], WIDE[2], WIDE[3], SUB[0], PLAIN[-2])
