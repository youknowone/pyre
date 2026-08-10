# CPython-suite gap: tuple consumers do not cover PyPy specialised-pair layouts.
# parity-tests reason: this targets PyPy/pyre pair storage and JIT consumers.

# Hot reads off an arity-2 tuple built by the interpreter, which routes
# through `makespecialisedtuple2` and so carries inline `value0` / `value1`
# slots instead of a `wrappeditems` array.  Every consumer that folds off the
# array-backed layout has to reach these slots too, and the shapes below are
# the ones where reading the wrong slot, or reading it with the wrong
# representation, produces a value the accumulator cannot cancel.

II = (11, 22)
FF = (1.5, 2.5)
OO = ("ab", "cd")
MIXED = (7, "x")
NESTED = ((1, 2), (3, 4))
TRIPLE = (11, 22, 33)


def const_index(rounds):
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + II[0] + II[1] * 2) & 0xFFFFFFFF
    return acc


def float_index(rounds):
    # Plain accumulation stays exact in a double and separates a slot swap
    # (2.5 * 3 + 1.5) from the right reading (1.5 * 3 + 2.5).
    acc = 0.0
    for i in range(rounds):
        acc = acc + FF[0] * 3 + FF[1]
    return acc


def object_index(rounds):
    # The object-slot read stays residual — its fold is wrong code and is
    # declined — so this covers the value the residual has to keep answering.
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + len(OO[0]) + len(OO[1]) * 2) & 0xFFFFFFFF
    return acc


def mixed_index(rounds):
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + MIXED[0] + len(MIXED[1])) & 0xFFFFFFFF
    return acc


def alternating_index(rounds):
    # The same subscript site sees both slots, so a fold that pins one index
    # has to side-exit rather than keep reading the pinned field.
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + II[i & 1]) & 0xFFFFFFFF
    return acc


def bool_index(rounds):
    # `True` is an int subclass and shares int's payload, so it is a valid
    # index and must select slot 1.
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + II[True] + II[False]) & 0xFFFFFFFF
    return acc


def negative_index(rounds):
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + II[-1] + II[-2] * 2) & 0xFFFFFFFF
    return acc


def out_of_range(rounds):
    acc = 0
    for i in range(rounds):
        try:
            acc = (acc * 31 + II[2]) & 0xFFFFFFFF
        except IndexError:
            acc = (acc * 31 + 9) & 0xFFFFFFFF
    return acc


def length(rounds):
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + len(II) + len(FF) * 2 + len(OO) * 3 + len(TRIPLE) * 5) & 0xFFFFFFFF
    return acc


def nested_index(rounds):
    # A pair whose slots are themselves pairs: the object slot read must
    # hand back a tuple that still subscripts correctly.
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + NESTED[0][1] * 5 + NESTED[1][0] * 2) & 0xFFFFFFFF
    return acc


def identity(rounds):
    # Small ints are cached, so a slot read has to produce the very object
    # the interpreter would have produced, not merely an equal one.
    eleven = 11
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + (1 if II[0] is eleven else 0)) & 0xFFFFFFFF
    return acc


def subclass_index(rounds):
    # A tuple subclass overriding `__getitem__` never uses a specialised
    # layout, and its override must still win.
    class Odd(tuple):
        def __getitem__(self, index):
            return 99

    odd = Odd((11, 22))
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + odd[0] + len(odd)) & 0xFFFFFFFF
    return acc


def escaping(rounds):
    kept = []
    acc = 0
    for i in range(rounds):
        item = II[i & 1]
        kept.append(item)
        acc = (acc * 31 + item) & 0xFFFFFFFF
    return (acc, kept[0], kept[1], len(kept))


def equality(rounds):
    acc = 0
    other_ii = (11, 22)
    other_ff = (1.5, 2.5)
    other_oo = ("ab", "cd")
    nan = float("nan")
    nan_pair = (nan, 1.0)
    acc_pairs = (
        (II, other_ii),
        (II, (11, 23)),
        (FF, other_ff),
        (OO, other_oo),
        (MIXED, (7, "x")),
        (II, TRIPLE),
        (nan_pair, nan_pair),
        (nan_pair, (nan, 1.0)),
    )
    for i in range(rounds):
        for a, b in acc_pairs:
            acc = (acc * 31 + (1 if a == b else 0) + (2 if a != b else 0)) & 0xFFFFFFFF
    return acc


def ordering(rounds):
    acc = 0
    for i in range(rounds):
        acc = (acc * 31
               + (1 if II < (11, 23) else 0)
               + (2 if II > (10, 99) else 0)
               + (4 if FF <= (1.5, 2.5) else 0)) & 0xFFFFFFFF
    return acc


def membership(rounds):
    acc = 0
    for i in range(rounds):
        acc = (acc * 31
               + (1 if 11 in II else 0)
               + (2 if 33 in II else 0)
               + (4 if "ab" in OO else 0)) & 0xFFFFFFFF
    return acc


for fn in (const_index, float_index, object_index, mixed_index,
           alternating_index, bool_index, negative_index, out_of_range,
           length, nested_index, identity, subclass_index, escaping,
           equality, ordering, membership):
    print(fn.__name__, fn(3000))
print("OK")
