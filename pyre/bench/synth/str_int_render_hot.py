# pyre-check: max-pypy-ratio=15
# pyre-check: spec-folds=str_call
# Hot-loop `str(int)`. The Python-level call has to reach the same decimal
# render the rtyper gives an unboxed `str(int)` -- one elidable
# `jit_int_str` over the unboxed operand -- instead of the opaque
# `bh_call_fn(str_type, NULL, i)` residual. That residual is a `CallMayForce`,
# so it clears the heap cache and forces virtualizables across itself; a loop
# built around it pays far more than the one string it allocates.
#
# `hot_render` is the speed leg, and it is written to carry the fold's own
# subject and nothing else: the checksum reads the rendered length, so the
# string has to materialize, but the loop performs no subscript. A subscript
# term would put an unrelated cost under the ceiling below -- a fixed string
# indexed in a loop measures 64 ns/iter here against CPython's 24, and this
# fold does not move it -- and the correctness legs are sized down for the
# same reason. `hot_render_signed` keeps a first-digit read, at a size that
# checks the sign digit without dominating the sum.
#
# The other legs are correctness legs for the shapes the fold must REFUSE,
# each written so a wrongly-admitted shape is a wrong number and not a silent
# pass:
#
#   * `bool` renders `True`/`False`, so admitting it through the `int` layout
#     it shares would shorten every term by three or four.
#   * an `int` SUBCLASS shares `ob_type == &INT_TYPE` and may override
#     `__str__`; admitting it drops the prefix its `__str__` adds.
#   * a `W_LongObject` keeps a pointer where `intval` sits, so admitting one
#     renders an address instead of the digits.
#   * a `str` SUBCLASS reboxes through its own `__new__`; admitting it hands
#     back a plain `str`, which the reported type name catches.
#
# `fresh_identity` is the leg for what the fold must not do to the renders it
# DOES admit. A render of more than one code point has storage identity under
# `is_w`, so two calls on the same operand are two objects; recording the call
# as elidable let the pure pass share one, and the loop below counts that
# sharing directly rather than inferring it from a timing.
#
# The ceiling is set BETWEEN the two arms of the fold, not at the measured
# number: `PYRE_FBW_NO_SPECIALIZE=str_call` reads 38.2x pypy and the default
# reads 11.5x (9.6x through check.py's own medians), so 15 catches the fold
# going away and still leaves room for a host that scales the two runners
# differently. It is a target, not a fit -- the remaining distance to pypy is
# the string allocation the fold still performs and pypy does not, so the
# number should come down again rather than be re-fitted upward.
#
# `spec-folds` is the other half and gates the subject exactly: a ceiling
# cannot tell a fold that stopped firing from a fixture nobody wrote a leg
# for, and it reads the same on every host.
#
# Deterministic; output asserted cpython==pypy.


class Prefixed(int):
    def __str__(self):
        return "P" + int.__repr__(self)


class MyStr(str):
    pass


def hot_render(n):
    acc = 0
    for i in range(n):
        acc = (acc * 31 + len(str(i))) & 0xFFFFFFFFFF
    return acc


def hot_render_signed(n, base):
    acc = 0
    for i in range(n):
        s = str(base - i)
        acc = (acc * 31 + ord(s[0]) + len(s)) & 0xFFFFFFFFFF
    return acc


def declined_shapes(n, big):
    acc = 0
    for i in range(n):
        acc += len(str(i % 2 == 0))
        acc += len(str(Prefixed(i)))
        acc += len(str(big + i))
    return acc


def fresh_identity(n, i):
    shared = 0
    for _ in range(n):
        if str(i) is str(i):
            shared += 1
        if repr(i) is repr(i):
            shared += 1
    return shared


def declined_callables(n):
    acc = 0
    names = 0
    for i in range(n):
        wrapped = MyStr(i)
        acc += len(wrapped)
        names += len(type(wrapped).__name__)
    return acc, names


def main():
    n = 6000000
    print("render", hot_render(n))
    # Negative values exercise the sign digit and the widest decimal the fold
    # can be asked for; INT_MIN has no positive counterpart, so a render that
    # negates before formatting reads wrong here rather than nowhere.
    print("signed", hot_render_signed(300000, -(1 << 62)))
    print("edges", str(-(1 << 63)), str((1 << 63) - 1), str(0), str(-1))
    print("declined", declined_shapes(20000, 1 << 70))
    print("callables", declined_callables(20000))
    print("shared", fresh_identity(20000, 123456))


main()
