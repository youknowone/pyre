# CPython-suite gap: nothing in the suite walks a traceback's frame locals in a
# loop tight enough for a minor collection to land between the walk and the read.
# parity-tests reason: this is a pyre/PyPy moving-GC regression gate.

"""A traceback walk reading `f_locals` must survive the collections it triggers.

On 2026-08-04 this body aborted 3/3 on cranelift with

    GC BUG: invalid type_id=<varies> at obj_addr=0x...
      (site=minor_varsize_item_target, holder_type_id=Some(9), holder_offset=Some(8))

The holder was a varsize object carrying its length (13) at offset 0 and its
items from offset 8, with only `items[0]` set, sitting above `nursery_top` --
so an old-gen holder pointing into the nursery. The reported `type_id` differed
every run and was pointer-shaped, meaning the target's header was recycled
memory: an unbarriered old-to-young store whose target an earlier minor
collection had already moved. Reached from `gc_alloc_nursery_shim` ->
`alloc_with_type_slow` -> `do_collect_nursery` -> `trace_and_update_object` ->
`copy_nursery_object`.

It is green now, on both backends, and this file is here so it stays that way
rather than because it still reproduces.

Re-measured 2026-08-30 on cranelift: 23 runs clean -- three with no environment,
two each at `PYPY_GC_NURSERY` 64/256/1536/3072/8192, and three each under
`MAJIT_GC_NURSERY_POISON=1` at the default, 1536 and 64. No nursery size is
pinned here because none of them discriminated.

The trigger was allocation layout rather than the loop bounds: the same body
reading `N`/`WHICH` from `sys.argv` was clean, and wrapping the defaults to
defeat constant folding did not change it.
"""

N = 15000
WHICH = "head"

ERR = ValueError("boom")


def mid(_i):
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
assert r == [(("drive", ("e", "k", "seen")), "mid")], r

print("OK")
