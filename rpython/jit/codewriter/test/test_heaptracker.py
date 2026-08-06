from rpython.jit.codewriter.heaptracker import get_fielddescr_index_in
from rpython.rtyper.lltypesystem import lltype


def test_get_fielddescr_index_in():
    S = lltype.GcStruct('S', ('a', lltype.Signed), ('b', lltype.Signed))
    assert get_fielddescr_index_in(S, 'a') == 0
    assert get_fielddescr_index_in(S, 'b') == 1
    assert get_fielddescr_index_in(S, 'missing') == -3

def test_get_fielddescr_index_in_leading_substruct():
    # the sub-struct starts at index 0, so the recursion is handed
    # cur_index == 0 and there is nothing to count twice
    INNER = lltype.Struct('INNER', ('x', lltype.Signed), ('y', lltype.Signed))
    S = lltype.GcStruct('S', ('inner', INNER), ('b', lltype.Signed))
    assert get_fielddescr_index_in(S, 'x') == 0
    assert get_fielddescr_index_in(S, 'y') == 1
    assert get_fielddescr_index_in(S, 'b') == 2
    assert get_fielddescr_index_in(S, 'missing') == -4

def test_get_fielddescr_index_in_substruct_after_a_field():
    # here the recursion is handed a nonzero cur_index, and the value it
    # returns already includes it.  Declaration order is a, x, y, b, so the
    # walk has to number them 0, 1, 2, 3 -- same as all_fielddescrs().
    INNER = lltype.Struct('INNER', ('x', lltype.Signed), ('y', lltype.Signed))
    S = lltype.GcStruct('S', ('a', lltype.Signed), ('inner', INNER),
                        ('b', lltype.Signed))
    assert get_fielddescr_index_in(S, 'a') == 0
    assert get_fielddescr_index_in(S, 'x') == 1
    assert get_fielddescr_index_in(S, 'y') == 2
    assert get_fielddescr_index_in(S, 'b') == 3
    assert get_fielddescr_index_in(S, 'missing') == -5

def test_get_fielddescr_index_in_two_substructs():
    # the second sub-struct is reached with cur_index == 2
    A = lltype.Struct('A', ('x', lltype.Signed), ('y', lltype.Signed))
    B = lltype.Struct('B', ('z', lltype.Signed))
    S = lltype.GcStruct('S', ('a', A), ('b', B), ('c', lltype.Signed))
    assert get_fielddescr_index_in(S, 'x') == 0
    assert get_fielddescr_index_in(S, 'y') == 1
    assert get_fielddescr_index_in(S, 'z') == 2
    assert get_fielddescr_index_in(S, 'c') == 3
    assert get_fielddescr_index_in(S, 'missing') == -5
