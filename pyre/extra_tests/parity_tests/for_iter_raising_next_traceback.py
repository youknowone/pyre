# CPython-suite gap: the suite never inspects the traceback of a
# non-StopIteration exception raised out of __next__ by a hot FOR_ITER.
# parity-tests reason: FOR_ITER's mismatch arm re-raises a value that already
# carries this frame's traceback node, so the re-raise must not record a second
# one.

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

import dis
import sys
import traceback


class Boom:
    def __iter__(self):
        return self

    def __next__(self):
        return self.missing


def enclosed():
    for value in Boom():
        return value


def bare():
    for value in Boom():
        return value


def for_iter_offset(function):
    return next(
        instruction.offset
        for instruction in dis.get_instructions(function)
        if instruction.opname == "FOR_ITER"
    )


# The looping frame's node must name the FOR_ITER that was in flight, not
# whatever instruction a coarser coordinate lookup happens to land on.  Derive
# the offset from the function's own bytecode so the check states the rule
# instead of pinning a layout.
expected_lasti = {
    "enclosed": for_iter_offset(enclosed),
    "bare": for_iter_offset(bare),
}

shapes = set()
loop_frame_lasti = set()


def observe(tb):
    shapes.add(tuple(f.name for f in traceback.extract_tb(tb)))
    while tb is not None:
        name = tb.tb_frame.f_code.co_name
        if name in expected_lasti:
            loop_frame_lasti.add((name, tb.tb_lasti))
        tb = tb.tb_next


for _ in range(8):
    try:
        try:
            enclosed()
        except AttributeError:
            raise
    except AttributeError:
        observe(sys.exc_info()[2])

    try:
        bare()
    except AttributeError:
        observe(sys.exc_info()[2])

assert shapes == {
    ("<module>", "enclosed", "__next__"),
    ("<module>", "bare", "__next__"),
}, sorted(shapes)
assert loop_frame_lasti == {
    ("enclosed", expected_lasti["enclosed"]),
    ("bare", expected_lasti["bare"]),
}, sorted(loop_frame_lasti)
print("OK")
