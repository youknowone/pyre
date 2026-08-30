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
# string has to materialize, but the loop performs no subscript.  That keeps
# unrelated string-indexing cost out of the fold census. `hot_render_signed`
# keeps a first-digit read at a size that checks the sign digit without
# dominating the sum.
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
# `spec-folds` gates the subject exactly: a residual produces the same strings,
# so output parity cannot tell a fold that stopped firing from a fixture nobody
# wrote a leg for.
#
# Deterministic; output asserted cpython==pypy.


class Prefixed(int):
    def __str__(self):
        return "P" + int.__repr__(self)


class MyStr(str):
    pass


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
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
    n = 100000
    print("render", hot_render(n))
    # Negative values exercise the sign digit and the widest decimal the fold
    # can be asked for; INT_MIN has no positive counterpart, so a render that
    # negates before formatting reads wrong here rather than nowhere.
    print("signed", hot_render_signed(5000, -(1 << 62)))
    print("edges", str(-(1 << 63)), str((1 << 63) - 1), str(0), str(-1))
    print("declined", declined_shapes(1000, 1 << 70))
    print("callables", declined_callables(1000))
    print("shared", fresh_identity(1000, 123456))


main()
