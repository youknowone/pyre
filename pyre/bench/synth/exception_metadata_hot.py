# Exception metadata observed from inside JIT-hot handlers: traceback
# attachment and shape, __context__ / __cause__ chaining, sys.exc_info, handler
# name clearing, and re-raise.
#
# The module also exercises the one-byte operand namespace of its own jitcode:
# each section adds ref registers and constants, and the constants routed
# through the per-kind pool sit ABOVE `count_regs[ref]` in the same byte
# (`assembler.py:131-137`). A module that overruns it must decline the graph
# (`assembler.py:265-269 check_result`), not abort while emitting the operand.

import sys

N = 3000


def out(key, value):
    print(f"{key} = {value}")


def thrower(i):
    raise KeyError(i)


def returns_through_finally(i):
    try:
        if i % 3 == 0:
            raise ValueError(i)
        return "ok"
    except ValueError:
        return "caught"
    finally:
        pass


# Same-frame raise/except: the handler sees a traceback, anchored at this
# frame and at one line.
same_frame_tb = 0
tb_linenos = set()
tb_frame_is_self = 0
for i in range(N):
    try:
        raise ValueError(i)
    except ValueError as e:
        traceback = e.__traceback__
        same_frame_tb += traceback is not None
        tb_linenos.add(traceback.tb_lineno)
        tb_frame_is_self += traceback.tb_frame is sys._getframe()
out("same_frame_tb", same_frame_tb)
out("same_frame_tb_lineno_count", len(tb_linenos))
out("same_frame_tb_frame_is_self", tb_frame_is_self)

# Cross-frame raise: the traceback spans the helper frame and this one.
cross_frame_depths = set()
for i in range(N):
    try:
        thrower(i)
    except KeyError as e:
        depth = 0
        traceback = e.__traceback__
        while traceback is not None:
            depth += 1
            traceback = traceback.tb_next
        cross_frame_depths.add(depth)
out("cross_frame_tb_depths", sorted(cross_frame_depths))

# Raising inside a handler chains exactly one __context__ link — an implicit
# chain that keeps growing would show up here as a larger set.
context_lengths = set()
for i in range(N):
    try:
        try:
            raise ValueError(i)
        except ValueError:
            raise TypeError(i)
    except TypeError as e:
        length = 0
        context = e.__context__
        seen = set()
        while context is not None and id(context) not in seen:
            seen.add(id(context))
            length += 1
            context = context.__context__
        context_lengths.add(length)
out("context_chain_lengths", sorted(context_lengths))

# `raise ... from ...` sets __cause__ and suppresses the context.
cause_ok = 0
for i in range(N):
    try:
        try:
            raise ValueError(i)
        except ValueError as cause:
            raise TypeError(i) from cause
    except TypeError as e:
        cause_ok += isinstance(e.__cause__, ValueError) and e.__suppress_context__
out("cause_ok", cause_ok)

# sys.exc_info reports the handled exception inside the handler and nothing
# once it has been left.
exc_info_inside = 0
exc_info_after_none = 0
for i in range(N):
    try:
        raise ValueError(i)
    except ValueError:
        info = sys.exc_info()
        exc_info_inside += info[0] is ValueError and info[1].args == (i,) and info[2] is not None
    exc_info_after_none += sys.exc_info() == (None, None, None)
out("exc_info_inside", exc_info_inside)
out("exc_info_after_none", exc_info_after_none)

# The handler binds the raised object itself and unbinds the name on exit.
raised_identity = 0
name_cleared = 0
for i in range(N):
    raised = ValueError(i)
    try:
        raise raised
    except ValueError as e:
        raised_identity += e is raised
    try:
        e
    except NameError:
        name_cleared += 1
out("raised_identity", raised_identity)
out("name_cleared", name_cleared)

# A bare re-raise keeps the original traceback rather than restarting it.
reraise_depths = set()
for i in range(N):
    try:
        try:
            thrower(i)
        except KeyError:
            raise
    except KeyError as e:
        depth = 0
        traceback = e.__traceback__
        while traceback is not None:
            depth += 1
            traceback = traceback.tb_next
        reraise_depths.add(depth)
out("reraise_tb_depths", sorted(reraise_depths))

# return-from-except under a finally.
finally_counts = {}
for i in range(N):
    result = returns_through_finally(i)
    finally_counts[result] = finally_counts.get(result, 0) + 1
out("finally_counts", sorted(finally_counts.items()))
