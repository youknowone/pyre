# An intersection walks the shorter side and keeps that side's objects, so
# when two equal elements are distinct objects which one survives depends on
# the operand lengths. The shortest operand seeds the result, measured as
# given -- a generator has no length and never seeds, and a list is measured
# with its duplicates. A warmup loop exercises the intersection path.
def warm(n):
    acc = 0
    for i in range(n):
        s = {0, 1, 2, 3}
        s.intersection_update([i % 4, (i + 1) % 4])
        acc += len(s) + len({0, 1, 2} & {1, 2, 3})
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def s(x):
    return sorted(x, key=repr)


def iu(a, b):
    a.intersection_update(b)
    return s(a)


def main():
    print("warm", warm(15000))
    # self longer than the other -> the other's object survives
    m("iu_int_float", lambda: iu({1, 2}, [1.0]))
    m("and_int_float", lambda: s({1, 2} & {1.0}))
    m("self2_other1", lambda: s({1, 2} & {1.0}))
    m("self3_other2", lambda: s({1, 2, 5} & {1.0, 2.0}))
    m("r_self1_other2", lambda: s({1.0, 2.0} & {1}))
    m("iu_set_rhs", lambda: iu({1, 2}, {1.0}))
    m("iu_frozen_rhs", lambda: iu({1, 2}, frozenset([1.0])))
    m("inter_int_float", lambda: s({1, 2}.intersection([1.0])))
    # self shorter than or equal to the other -> self's object survives
    m("iu_float_int", lambda: iu({1.0, 2}, [1]))
    m("and_float_int", lambda: s({1.0, 2} & {1}))
    m("self1_other2", lambda: s({1} & {1.0, 2.0}))
    m("self2_other3", lambda: s({1, 2} & {1.0, 2.0, 5.0}))
    m("inter_float_int", lambda: s({1.0, 2}.intersection([1])))
    m("rand_int_float", lambda: s(frozenset([1, 2]) & {1.0}))
    # a generator has no length, so it never seeds even when it is shortest;
    # the result then outgrows it and the walk swaps back to the generator
    m("gen_other", lambda: s({1, 2}.intersection(x for x in [1.0])))
    m("gen_only_self2", lambda: s({1.0, 2.0}.intersection(x for x in [1])))
    # a list is measured with its duplicates, so [1.0, 1.0] does not seed
    m("dup_list_len2", lambda: s({1, 2}.intersection([1.0, 1.0])))
    m("dup_list_len3", lambda: s({1, 2}.intersection([1.0, 1.0, 1.0])))
    # multiple operands: the globally shortest seeds, not the first other
    m("multi_int_float", lambda: s({1, 2, 3}.intersection({1.0, 2.0, 3.0}, {1.0})))
    m("multi_self_smallest", lambda: s({1}.intersection({1.0, 2.0}, {1.0, 3.0})))
    m("multi_smallest_middle", lambda: s({1, 2, 3}.intersection([1.0, 2.0], [1.5, 1.0])))
    # the other operations keep self's object
    m("union_int_float", lambda: s({1, 2} | {1.0}))
    m("union_float_int", lambda: s({1.0, 2} | {1}))
    m("sym_int_float", lambda: s({1, 2} ^ {1.0, 3}))
    m("update_int_float", lambda: (lambda a: (a.update([1.0]), s(a))[1])({1, 2}))
    m("add_float_over_int", lambda: (lambda a: (a.add(1.0), s(a))[1])({1, 2}))
    m("add_int_over_float", lambda: (lambda a: (a.add(1), s(a))[1])({1.0, 2}))
    # an unhashable element raises even from an operand that cannot seed
    m("unhashable_other", lambda: {1, 2}.intersection([[]]))
    m("unhashable_gen", lambda: {1, 2}.intersection(x for x in [[]]))
    # ordinary intersections are unchanged
    m("empty_other", lambda: s({1, 2}.intersection([])))
    m("empty_self", lambda: s(set().intersection([1.0])))
    m("no_args", lambda: s({1, 2}.intersection()))
    m("str_other", lambda: s(set("abc").intersection("ab")))
    m("iu_two_others", lambda: iu({1, 2}, [1.0]))
    # the result class follows the left operand
    m("frozen_self_class", lambda: type({1} & frozenset([1])).__name__)
    m("frozen_lhs_class", lambda: type(frozenset([1]) & {1}).__name__)
    m("inter_frozen_class", lambda: type(frozenset([1]).intersection([1])).__name__)


main()
