# pyre-check: max-pypy-ratio=47
# An exception raised in an inlined callee and caught two or three frames up
# must still attach a traceback node for every frame it passed through: one at
# the raise instruction and one per callee->caller propagation.
#
# The callee frames are the exposed ones.  A trace's own frame is a real
# PyFrame, so the `exit_frame_with_exception` it finishes with surfaces as an
# error there and `handle_operation_error` attaches that node.  An inlined
# callee is popped symbolically by the walk and never becomes a frame, so
# unless the trace records the node itself the chain comes out one (or, for the
# three-deep shape, two) frames short once the loop runs compiled -- while the
# pre-compile iterations stay correct, which is why only the per-iteration
# shape set catches it.
#
# Shapes: explicit `raise` two frames down, a builtin ZeroDivisionError two
# frames down, and an explicit `raise` three frames down.
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


def catch_two_frames(i):
    try:
        raise_mid(i)
    except ValueError as e:
        return frame_names(e.__traceback__)


def divide_leaf(i):
    return i // 0


def divide_mid(i):
    return divide_leaf(i)


def catch_builtin_two_frames(i):
    try:
        divide_mid(i)
    except ZeroDivisionError as e:
        return frame_names(e.__traceback__)


def deep_leaf(i):
    raise TypeError(i)


def deep_mid(i):
    deep_leaf(i)


def deep_top(i):
    deep_mid(i)


def catch_three_frames(i):
    try:
        deep_top(i)
    except TypeError as e:
        return frame_names(e.__traceback__)


def survey(label, fn):
    shapes = set()
    for i in range(N):
        shapes.add(fn(i))
    print(label, sorted(shapes))


survey("two_frames   =", catch_two_frames)
survey("builtin_two  =", catch_builtin_two_frames)
survey("three_frames =", catch_three_frames)
