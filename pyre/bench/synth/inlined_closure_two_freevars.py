# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=defined_in_the_loop,made_by_a_factory,a_two_name_lambda
# An inlined callee closing over exactly TWO names answered with the `cell`
# CLASS where its first freevar belonged, on both backends, every run:
#
#     TypeError: unsupported operand type(s) for +: 'type' and 'int'
#
# ## What it was
#
# A closure over two names carries a two-element `__closure__`, and
# `w_tuple_new` routes every arity-2 tuple to a specialised variant.  Cells are
# objects, so that variant is `W_SpecialisedTupleObject_oo`, which stores its
# two refs inline and has no `wrappeditems` at all:
#
#     W_TupleObject                { ob_header(16), hash(8), wrappeditems @ 24 }
#     W_SpecialisedTupleObject_oo  { ob_header(16), hash(8), value0      @ 24,
#                                                            value1      @ 32 }
#
# The inlined-callee closure read emitted `GetfieldGcR(closure, wrappeditems)`
# with no class guard in front of it, so on an `_oo` it returned `value0` --
# the FIRST CELL -- and the two `GetarrayitemGcPureR` reads then indexed that
# cell as an `ItemsBlock` (`{ capacity @ 0, items @ 8 }`):
#
#     items[0] = *(cell + 8)  = the cell's `w_class`  -> <class 'cell'>
#     items[1] = *(cell + 16) = the cell's contents   -> the FIRST variable
#
# so the callee's freevars came out shifted by one.  `is_tuple` and
# `is_exact_tuple` are both type predicates and answer true for all three
# specialised variants, which is why nothing upstream of the read caught it.
# `bh_load_deref_value_fn` passes a non-cell in a freevar slot straight through
# rather than rejecting it, which is what turned the mis-read into a wrong
# answer several frames away instead of a failure at the slot.
#
# ## Why the shapes are exactly these
#
# The specialisation is arity-2 only, and that is the whole condition: a callee
# closing over 1, 3 or 4 names passed (3 swept n=1500..40000).  `n` has to
# clear the trace threshold -- 1000 passed, 1500 and up failed.
#
# `defined_in_the_loop` is the minimal witness.  `made_by_a_factory` is the one
# that proves the reach: with the closure built in another function there is no
# `NewArrayClear(2)` in the traced loop at all, so the heap cache is not
# involved and an ordinary decorator or callback is in range.
# `a_two_name_lambda` covers the arm where `co_freevars` is not in source
# order.  Each of the three failed on its own before the fix.
EXPECTED = (9000000, 17997000, 36003000)


def make(a, b):
    def add():
        return (a, b)

    return add


def defined_in_the_loop(n):
    t = 0
    for i in range(n):
        x = i
        y = i + 1

        def inner():
            return x + y

        t += inner()
    return t


def made_by_a_factory(n):
    t = 0
    for i in range(n):
        add = make(i, i + 1)
        a, b = add()
        t += a * 3 + b
    return t


def a_two_name_lambda(n):
    t = 0
    for i in range(n):
        lo = i
        hi = i + 5
        g = lambda: (lo, hi)
        p, q = g()
        t += p * 7 + q
    return t


got = (defined_in_the_loop(3000), made_by_a_factory(3000), a_two_name_lambda(3000))
if got == EXPECTED:
    print("PASS inlined closure two freevars")
else:
    print("FAIL %r against %r" % (got, EXPECTED))
    raise SystemExit(1)
