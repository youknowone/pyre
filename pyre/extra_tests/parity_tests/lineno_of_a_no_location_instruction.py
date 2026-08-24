# CPython-suite gap: `test_traceback`, `test_frame` and `test_sys_settrace` all
# read line numbers, but only off instructions the compiler gave a position to.
# Nothing in the suite stops on one of the `NO_LOCATION` ranges a `with` or a
# `try`/`finally` cleanup sits in, so a runtime that answered those with the
# line that happened to precede them would pass every module.
#
# parity-tests reason: `co_positions` already reports those instructions with a
# `None` line, so the information is present; what is observable is whether the
# two readers of a *resolved* line agree with it.  `PyCode_Addr2Line` answers
# `-1` there, and both `tb_lineno_get` and `frame_getlineno` turn a line below
# zero into `None`.  A runtime that expands the line table into one entry per
# instruction cannot represent the gap and answers the previous line instead --
# the same expansion also loses a zero `co_firstlineno`, which the constructor
# accepts, so both are pinned here.
#
# PyPy 7.3.20 is a 3.11 line table and has no `TracebackType` sentinel, so it
# fails the rebuilt cases for the reason `traceback_lineno_sentinel.py` gives
# rather than for anything about a missing line.

import sys
import types

SOURCE = '''
def guarded(cm):
    with cm:
        raise ValueError('body')
'''

NAMESPACE = {}
exec(compile(SOURCE, '<no-location>', 'exec'), NAMESPACE)
GUARDED = NAMESPACE['guarded']
POSITIONS = list(GUARDED.__code__.co_positions())
NO_LOCATION = [index for index, row in enumerate(POSITIONS) if row[0] is None]


FRAMES = []


class Manager:
    def __enter__(self):
        FRAMES.append(sys._getframe(1))
        return self

    def __exit__(self, *unused):
        return False


try:
    GUARDED(Manager())
except ValueError:
    pass


def the_cleanup_of_a_with_carries_no_line():
    # Derived rather than written out: the count is what matters, not which
    # offsets the compiler put them at.
    print('no-location instructions:', len(NO_LOCATION) > 0)
    assert NO_LOCATION, POSITIONS


def a_traceback_rebuilt_on_one_reports_no_line():
    # A live frame for that code object, so the rebuilt node resolves against
    # the same line table.  Reached through `__enter__` rather than a trace
    # function on purpose: `PyCode_Addr2Line` consults the monitoring line
    # table ahead of `co_linetable` once a code object is instrumented, and
    # that one answers only at instruction starts, so tracing this frame would
    # report `None` for every inline cache as well.
    frame = FRAMES[0]
    TracebackType = types.TracebackType
    lines = [TracebackType(None, frame, index * 2, -1).tb_lineno for index in NO_LOCATION]
    print('tb_lineno on no-location:', lines)
    assert lines == [None] * len(NO_LOCATION), lines
    # The control: every instruction that does carry a line still answers one,
    # so the `None`s above are the gap and not a resolver that stopped early.
    carried = [index for index, row in enumerate(POSITIONS) if row[0] is not None]
    answered = [TracebackType(None, frame, index * 2, -1).tb_lineno for index in carried]
    print('every carrying instruction answers a line:',
          all(line is not None for line in answered))
    assert all(line is not None for line in answered), answered


def a_frame_stopped_on_one_reports_no_line():
    seen = {}
    reprs = {}

    def tracer(frame, event, arg):
        if frame.f_code is GUARDED.__code__:
            frame.f_trace_opcodes = True
            if event == 'opcode':
                index = frame.f_lasti // 2
                if index in NO_LOCATION:
                    seen[index] = frame.f_lineno
                    reprs[index] = repr(frame).split(', line ')[1].split(',')[0]
        return tracer

    sys.settrace(tracer)
    try:
        GUARDED(Manager())
    except ValueError:
        pass
    finally:
        sys.settrace(None)
    executed = sorted(seen)
    print('f_lineno on no-location:', [seen[index] for index in executed])
    print('frame repr line there:', sorted(set(reprs.values())))
    assert executed, (seen, NO_LOCATION)
    assert all(seen[index] is None for index in executed), seen


def holder():
    HOLDER.append(sys._getframe())
    return 1


HOLDER = []
holder()
RESTING = HOLDER[0]


def a_frame_for(linetable):
    """A frame of `holder` recompiled with a replacement line table."""
    replaced = holder.__code__.replace(co_linetable=linetable)
    types.FunctionType(replaced, globals())()
    return HOLDER[-1]


def a_line_table_covering_nothing_resolves_to_none():
    frame = a_frame_for(b'')
    TracebackType = types.TracebackType
    lines = [TracebackType(None, frame, lasti, -1).tb_lineno for lasti in (0, 2, 4, 100)]
    print('empty line table:', lines, frame.f_lineno)
    assert lines == [None, None, None, None], lines
    assert frame.f_lineno is None, frame.f_lineno


def a_line_table_covering_one_instruction_stops_there():
    # One header byte with a zero payload code: a single two-byte range on the
    # first line, and nothing after it.
    frame = a_frame_for(b'\x00')
    firstlineno = holder.__code__.co_firstlineno
    TracebackType = types.TracebackType
    lines = [TracebackType(None, frame, lasti, -1).tb_lineno for lasti in (0, 2, 4, 100)]
    print('one-range line table:', [lines[0] == firstlineno] + lines[1:])
    assert lines[0] == firstlineno, (lines[0], firstlineno)
    assert lines[1:] == [None, None, None], lines


def a_zero_first_line_is_the_answer_for_a_negative_offset():
    replaced = holder.__code__.replace(co_firstlineno=0)
    types.FunctionType(replaced, globals())()
    frame = HOLDER[-1]
    TracebackType = types.TracebackType
    lines = [TracebackType(None, frame, lasti, -1).tb_lineno for lasti in (-1, -2, 0)]
    print('zero co_firstlineno:', lines)
    assert lines == [0, 0, 0], lines


the_cleanup_of_a_with_carries_no_line()
a_traceback_rebuilt_on_one_reports_no_line()
a_frame_stopped_on_one_reports_no_line()
a_line_table_covering_nothing_resolves_to_none()
a_line_table_covering_one_instruction_stops_there()
a_zero_first_line_is_the_answer_for_a_negative_offset()
print('OK')
