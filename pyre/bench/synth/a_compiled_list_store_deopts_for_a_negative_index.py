# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=store_hot,read_hot
# A store recorded from a non-negative index must still deopt when a later
# index is NEGATIVE.  `space.setitem` remaps a negative index to `index + len`
# (listobject.py), so a direct items-block store that only proved
# `raw_index < len` addresses before the start of the array: the write is lost
# and it lands on the word ahead of the items, not on `lst[-1]`.
#
# THE ASSERTION IS TWO-SIDED ON PURPOSE.  Checking only that `lst[-1]` reads
# back the value passes even when the store went out of range, because the read
# arm has its own lower-bound guard and deopts.  The neighbours must be checked
# too, and so must the tail: an out-of-range write lands on the items block's
# own header, which a later `append` reads.
#
# The read arms are covered here as well so that the list, tuple and store
# arms cannot drift apart -- all three take the same shared bounds guard.
WARM = 6000  # past the loop threshold (1039) many times over
NEG = 5  # enough re-entries to reach the compiled store, not just the prologue


def store_hot(n, seq, idx, mkval):
    for i in range(n):
        seq[idx] = mkval(i)
    return seq


def read_hot(n, seq, idx):
    total = 0
    for _ in range(n):
        total += seq[idx]
    return total


CASES = (
    # (name, fresh list, mkval, readable)
    ('object', lambda: ['a', 'b', 'c', 'd'], lambda i: str(i), False),
    ('int', lambda: [0, 0, 0, 0], lambda i: i, True),
    ('float', lambda: [0.0, 0.0, 0.0, 0.0], lambda i: i * 1.0, True),
)


def check_store(name, fresh, mkval, failures):
    seq = fresh()
    store_hot(WARM, seq, 1, mkval)  # compile the store at a non-negative index
    untouched = [seq[0], seq[2]]
    # Re-enter the SAME trace: same callable (a fresh one would side-exit on the
    # callable guard before reaching the store) and enough iterations to get
    # past the entry threshold, with only the index sign changed.
    store_hot(NEG, seq, -1, mkval)
    want = mkval(NEG - 1)
    if seq[3] != want:
        failures.append('%s: lst[-1] left index 3 as %r, expected %r — the '
                        'store did not remap the negative index'
                        % (name, seq[3], want))
    if [seq[0], seq[2]] != untouched:
        failures.append('%s: neighbours moved to %r, expected %r'
                        % (name, [seq[0], seq[2]], untouched))
    if len(seq) != 4:
        failures.append('%s: len is %d after the negative store, expected 4'
                        % (name, len(seq)))
    # An out-of-range write lands on the items block's header, so the next
    # growth reads a corrupted capacity.
    seq.append(mkval(1))
    if len(seq) != 5 or seq[3] != want:
        failures.append('%s: append gave %r, expected the tail intact and len 5'
                        % (name, seq))
    # Both bounds must still raise rather than take the direct store.
    for bad in (len(seq), -len(seq) - 1):
        try:
            store_hot(NEG, seq, bad, mkval)
        except IndexError:
            pass
        else:
            failures.append('%s: lst[%d] = v did not raise IndexError' % (name, bad))


def check_read(name, fresh, mkval, failures):
    seq = fresh()
    for i in range(4):
        seq[i] = mkval(i + 1)
    for target, kind in ((seq, ''), (tuple(seq), ' tuple')):
        read_hot(WARM, target, 1)  # compile the read at a non-negative index
        for idx in (-1, -4):
            got = read_hot(NEG, target, idx)
            want = NEG * target[len(target) + idx]
            if got != want:
                failures.append('%s%s: [%d] read %r over %d iterations, '
                                'expected %r'
                                % (name, kind, idx, got, NEG, want))


def main():
    failures = []
    for name, fresh, mkval, readable in CASES:
        check_store(name, fresh, mkval, failures)
        if readable:
            check_read(name, fresh, mkval, failures)
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a compiled list store deopts for a negative index')
    return 0


import sys

sys.exit(main())
