# pyre-check: max-pypy-ratio=8
# startswith/endswith convert their code-point bounds to byte offsets: a
# start past the end is not clamped, it inverts the window, so the match is
# False even for an empty prefix. A start exactly at the end still yields a
# valid empty window and matches. Shifting zero left never allocates, however
# large the count. A warmup loop exercises the bounded prefix path.
#
# The bound varies per iteration so the match is executed rather than
# constant-folded: `rstring.py startswith` is `@jit.elidable`, and over
# constant operands both pypy and pyre fold the call away, leaving the loop
# measuring nothing but its own counter.
#
# The count is what keeps the ratio gate armed -- under it pypy's
# execution-only time subtracts to the floor, check.py declines both bounds
# on a clamped baseline, and the recorded ceiling goes unapplied. It is
# bounded from above too: cpython runs this loop interpreted, and a fixture
# whose cpython reference exceeds its 5s timeout loses the cpython/pypy
# output cross-check.
def warm(n):
    acc = 0
    for i in range(n):
        start = i % 3
        if "abcdef".startswith("cd", start, 4):
            acc += 1
        if "abcdef".endswith("ef", start, 6):
            acc += 1
        acc += 0 << (i % 8)
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def main():
    print("warm", warm(51612904))
    # a start past the end inverts the window: False even for an empty prefix
    m("sw_empty_oor", lambda: "abc".startswith("", 5, 10))
    m("ew_empty_oor", lambda: "abc".endswith("", 5, 10))
    m("sw_empty_oor_noend", lambda: "abc".startswith("", 5))
    m("ew_empty_inverted", lambda: "".endswith("", 1, 0))
    m("sw_oor_start", lambda: "abc".startswith("a", 5, 10))
    m("bytes_sw_empty_oor", lambda: b"abc".startswith(b"", 5, 10))
    m("bytes_ew_empty_oor", lambda: b"abc".endswith(b"", 5, 10))
    # a start exactly at the end is a valid empty window
    m("sw_empty_at_len", lambda: "abc".startswith("", 3, 10))
    m("ew_empty_at_len", lambda: "abc".endswith("", 3, 10))
    m("sw_empty_in", lambda: "abc".startswith("", 1, 2))
    m("sw_empty_zero", lambda: "abc".startswith("", 0, 0))
    # an inverted window from start > end
    m("sw_start_gt_end", lambda: "abc".startswith("", 2, 1))
    m("sw_start_gt_end_needle", lambda: "abc".startswith("b", 2, 1))
    # ordinary bounded matches are unchanged
    m("sw_bounded", lambda: "abcdef".startswith("cd", 2, 4))
    m("ew_bounded", lambda: "abcdef".endswith("cd", 2, 4))
    m("sw_neg_start", lambda: "abcdef".startswith("ef", -2))
    m("ew_neg_end", lambda: "abcdef".endswith("cd", 0, -2))
    # a long is the int class with a bigint payload; the bound goes through
    # the slice-index conversion, not a machine-int unbox
    m("sw_neg_long", lambda: "abc".startswith("a", -(10**100)))
    m("sw_pos_long", lambda: "abc".startswith("a", 10**100))
    m("ew_neg_long", lambda: "abc".endswith("c", 0, -(10**100)))
    m("ew_pos_long", lambda: "abc".endswith("c", 0, 10**100))
    m("sw_no_bounds", lambda: "abc".startswith("ab"))
    m("sw_unicode", lambda: "éèx".startswith("", 5, 10))
    m("sw_unicode_ok", lambda: "éèx".startswith("è", 1, 2))
    # zero shifted left never allocates, however large the count
    m("zero_shift_huge", lambda: 0 << 10**18)
    m("zero_shift_big", lambda: 0 << (2**62))
    m("zero_shift_small", lambda: 0 << 5)
    m("zero_shift_zero", lambda: 0 << 0)
    m("one_shift_zero", lambda: 1 << 0)
    m("int_shift_ok", lambda: 1 << 10)


main()
