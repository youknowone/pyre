# startswith/endswith convert their code-point bounds to byte offsets: a
# start past the end is not clamped, it inverts the window, so the match is
# False even for an empty prefix. A start exactly at the end still yields a
# valid empty window and matches. Shifting zero left never allocates, however
# large the count. A warmup loop exercises the bounded prefix path.
def warm(n):
    acc = 0
    for i in range(n):
        if "abcdef".startswith("cd", 2, 4):
            acc += 1
        if "abcdef".endswith("ef", 0, 6):
            acc += 1
        acc += 0 << (i % 8)
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def main():
    print("warm", warm(15000))
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
