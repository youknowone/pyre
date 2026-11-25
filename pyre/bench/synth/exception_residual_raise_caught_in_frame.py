# pyre-check: max-pypy-ratio=11
# pyre-check: jitstats-band=guard_failures=24
# guard_failures follows the minor-collection schedule: 600 on darwin arm64 at
# check.py's 4MB PYPY_GC_NURSERY pin (5/5 runs), 616 at PYPY_GC_NURSERY=4536K
# and 619 at 4539K, with 4 loops / 2 bridges unchanged. CI macos-latest
# cranelift reads 616 and ubuntu cranelift 619 at the same pin.
# ubuntu dynasm 3.9x, cranelift 9.3x; floor 2.75x.
# A frame that catches an exception a residual call raised contributes its own
# traceback node, including once its `except` is reached from inside its own
# compiled trace.
#
# walk()'s SubRaise arm scans the raising frame for a `catch_exception/L` and
# jumps straight into the handler, so the handler ends up inside the trace and
# the frame never surfaces an error the interpreter could record a node from.
# Unless the trace carries the record itself, the chain comes out one frame
# short once the loop is compiled while the pre-compile iterations stay correct.
#
# The raise sites are builtin containers, so each one is a residual call rather
# than an inlined callee.
N = 145455


def names(traceback):
    out = []
    while traceback is not None:
        out.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    return tuple(out)


def missing_key(d, i):
    try:
        return d[i]
    except KeyError as e:
        return names(e.__traceback__)


def bad_index(seq, i):
    try:
        return seq[i]
    except IndexError as e:
        return names(e.__traceback__)


def bad_convert(s):
    try:
        return int(s)
    except ValueError as e:
        return names(e.__traceback__)


d = {}
seq = []
key_shapes = set()
index_shapes = set()
convert_shapes = set()
for i in range(N):
    key_shapes.add(missing_key(d, i))
    index_shapes.add(bad_index(seq, i))
    convert_shapes.add(bad_convert("x"))
print("key    ", sorted(key_shapes))
print("index  ", sorted(index_shapes))
print("convert", sorted(convert_shapes))
