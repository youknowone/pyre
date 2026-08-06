# pyre-check: max-pypy-ratio=53
# The ceiling is twice the slowest ratio observed, 26.2x on the macos
# runner; the runners read this fixture between 3.0x and 26.2x.
# The frame holding the `try` contributes its own traceback node, including
# when its `except` is reached from inside its own compiled trace.
#
# When an inlined callee raises, the caller-side handler scan routes the raise
# straight to the caller's `except` and the walk keeps going, so the handler
# ends up inside the trace. Compiled, the trace then catches the exception
# itself: the frame never surfaces an error to the interpreter, so unless the
# trace records the node it is the recording pass alone that ever applies it
# and the chain comes out missing its outermost frame.
#
# The dict lookup in the loop is load-bearing: it keeps the module-level loop
# from completing a trace of its own, so `shape` is compiled as a func-entry
# trace instead of being inlined into the caller.
N = 4000


def frame_names(traceback):
    names = []
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    return tuple(names)


def raise_leaf(i):
    raise ValueError(i)


def raise_mid(i):
    raise_leaf(i)


def shape(i):
    try:
        raise_mid(i)
    except ValueError as e:
        return frame_names(e.__traceback__)


seen = {}
shapes = set()
for i in range(N):
    shapes.add(shape(i))
    seen.get(0, 0)
print(sorted(shapes))
