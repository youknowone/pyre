"""Deterministic witness for `GC BUG ... site=minor_varsize_item_target`.

Run with no arguments. cranelift aborts 3/3; dynasm, `PYRE_NO_JIT=1`, pypy and
cpython all print `1 [(('drive', ('e', 'k', 'seen')), 'mid')]`.

OPEN, and independent of the traceback-journal and virtual-ref root-walk fixes:
it reproduces identically with either of those reverted.

    GC BUG: invalid type_id=<varies per run> at obj_addr=0x...fc28
      (header_addr=0x...fc20, nursery_start=0x...b0000,
       site=minor_varsize_item_target, nursery_free=0x...fee0,
       nursery_top=0x...b0000, holder_addr=0x...,
       holder_type_id=Some(9), holder_offset=Some(8),
       holder_words=[0xd, 0x...fc28, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0])

Reached from `gc_alloc_nursery_shim` -> `alloc_with_type_slow` ->
`do_collect_nursery` -> `trace_and_update_object` -> `copy_nursery_object`, so
it is a minor collection triggered by a nursery allocation out of compiled code.

What the message says: the holder is a varsize object of type_id 9 carrying its
length (13) at offset 0 and items from offset 8, and only `items[0]` is set. The
holder address is above `nursery_top`, so the holder is old-gen and `items[0]`
points into the nursery. The reported `type_id` differs every run and is
pointer-shaped, so the target's header is recycled memory rather than a live
object — the shape of an unbarriered old-to-young store whose target an earlier
minor collection already moved.

The trigger is allocation layout, not the loop bounds. The same body reading
`N`/`WHICH` from `sys.argv` is clean when `15000 head` is passed, and aborts when
it takes the identical values from `else` defaults; wrapping those defaults to
defeat constant folding (`int("15000")`, `len(sys.argv)` arithmetic) does not
change it, which is what rules the folding explanation out.
"""

N = 15000
WHICH = "head"

ERR = ValueError("boom")


def mid(i):
    raise ValueError("boom")


def locs(tb):
    out = []
    idx = 0
    while tb is not None:
        f = tb.tb_frame
        want = (
            WHICH == "all"
            or (WHICH == "head" and idx == 0)
            or (WHICH == "tail" and tb.tb_next is None)
        )
        if want:
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
            seen.add(locs(e.__traceback__))
            e.__traceback__ = None
        k += 1
    return sorted(seen)


r = drive()
print(len(r), r)
