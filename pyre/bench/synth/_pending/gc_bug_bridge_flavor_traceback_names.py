# KNOWN FAILING on cranelift: aborts with
#   GC BUG: invalid type_id=... site=object_total_size
# from `gc_alloc_nursery_shim` -> `alloc_with_type` -> `do_collect_nursery` ->
# `incremental_mark_step`, i.e. a dangling nursery pointer on the MAJOR gray
# stack.  dynasm and PYRE_JIT=0 are clean, which matches the known
# cranelift/wasm/Windows-only shape of that signature.
#
# Intermittent but high-rate - 9/10 runs - and needs no GC stress build.
#
# Ingredients, each of which removing made it stop aborting:
#   * TWO exception classes raised into one hot try/except, so a bridge is
#     compiled for the exception edge (the `exc_mixed_classes_bridge_flavor`
#     shape);
#   * an INLINED intermediate frame between the loop and the raise;
#   * reading `tb_frame.f_code.co_name` off every node in the handler;
#   * retaining the resulting name tuples in a set across iterations.
# Counting traceback depth instead of reading `co_name`, or dropping the set,
# or a single exception class, all run clean.
#
# Expected output: A [('T', 'a_bridge_two_classes', 'mid_two', 'leaf_two'),
#                     ('V', 'a_bridge_two_classes', 'mid_two', 'leaf_two')]
N = 60000


def chain(e):
    names = []
    tb = e.__traceback__
    while tb is not None:
        names.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    return tuple(names)


def leaf_two(i):
    if i % 3 == 1:
        raise ValueError(i)
    if i % 3 == 2:
        raise TypeError(i)
    return i


def mid_two(i):
    return leaf_two(i)


def a_bridge_two_classes():
    shapes = set()
    acc = 0
    for i in range(N):
        try:
            acc += mid_two(i)
        except ValueError as e:
            shapes.add(("V",) + chain(e))
        except TypeError as e:
            shapes.add(("T",) + chain(e))
    return sorted(shapes)


print("A", a_bridge_two_classes())
