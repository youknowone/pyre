# One exception object raised over and over, caught in a hot loop whose handler
# forces the catching frame through the traceback (`tb_frame.f_locals`). The
# force aborts the recording walk, and a walk that does not commit must undo the
# traceback node it already attached to the live exception — otherwise the
# interpreter's replay records the same frames a second time and the reused
# object hands out a chain of four nodes instead of two. A freshly raised
# exception hides this: the discarded walk's object is unreachable, so only a
# reused one accumulates both deliveries.
#
# N is bounded rather than sized for throughput. The doubling reproduces from
# roughly the 1200th iteration on both native backends, while cranelift takes a
# GC BUG (`minor_varsize_item_target`) on this shape somewhere between 4000 and
# 8000 — a pre-existing crash this workload happens to reach, not something the
# traceback chain is party to.
#
# No `max-pypy-ratio`: this is a traceback-shape oracle, and its pypy arm runs
# well under a tenth of a second — below the point where a wall-clock ratio
# measures the workload rather than process startup.
#
# Expected output: [(('drive', ('e', 'k', 'seen')), 'mid')]
N = 4000
ERR = ValueError("boom")


def mid(i):
    raise ERR


def shape(tb):
    out = []
    idx = 0
    while tb is not None:
        f = tb.tb_frame
        if idx == 0:
            out.append((f.f_code.co_name, tuple(sorted(f.f_locals))))
        else:
            out.append(f.f_code.co_name)
        tb = tb.tb_next
        idx += 1
    return tuple(out)


def drive():
    seen = set()
    k = 0
    while k < N:
        try:
            mid(k)
        except ValueError as e:
            seen.add(shape(e.__traceback__))
            e.__traceback__ = None
        k += 1
    return sorted(seen)


print(drive())
