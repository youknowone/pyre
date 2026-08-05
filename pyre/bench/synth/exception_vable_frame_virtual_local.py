# pyre-check: max-pypy-ratio=24
# Regression guard: when a traced function raises, the write-back into its own
# virtualizable frame must store every local, including the ones whose value is
# still a virtual box.
#
# `branchy` gets hot enough to be traced from its entry, so its frame is the
# trace's virtualizable and its locals live in registers. The raise exits the
# trace, and the exit emits `gen_store_back_in_vable` — a `SETFIELD_GC` per
# static field plus a `SETARRAYITEM_GC` per slot of `locals_cells_stack_w`
# (`rpython/jit/metainterp/pyjitpl.py:3489-3520`). `marker` holds a virtual
# `W_IntObject`, so its slot store was the one dropped at the heap optimizer's
# flush; `i` and `odd_only` came from input args and survived. The traceback
# node then names a frame whose only bound slot is the parameter.
#
# The conditional binding is what makes `branchy` hot as a function rather than
# folded into the driver's loop, and the run has to pass the function-entry
# threshold: the interpreted iterations write the frame directly and read
# correctly either way, so a short run cannot see this. Before the fix, dynasm
# and cranelift both lost `marker` from the iteration the entry trace compiled
# onwards.
#
# Expected output: 2000 ('i', 'marker') and 2000 ('i', 'marker', 'odd_only')

N = 4000


def branchy(i):
    marker = i * 2
    if i % 2:
        odd_only = marker
    raise ValueError(marker)


def drive():
    kinds = {}
    k = 0
    while k < N:
        try:
            branchy(k)
        except ValueError as e:
            tb = e.__traceback__.tb_next
            names = tuple(sorted(tb.tb_frame.f_locals)) if tb is not None else None
            kinds[names] = kinds.get(names, 0) + 1
        k += 1
    return kinds


for names, count in sorted(drive().items(), key=lambda kv: -kv[1]):
    print(count, names)
