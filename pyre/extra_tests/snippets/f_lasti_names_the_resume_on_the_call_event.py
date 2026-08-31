# pyre-check: gate=1
# pyre-check: pypy-diverges: `pyframe.py:78` declares `last_instr = -1` and
# `pyframe.py:335-341` states "Execution starts just after the last_instr.
# Initially, last_instr is -1", so `fget_f_lasti` (`pyframe.py:771`) hands out
# `-1` for a frame that has not run an instruction.  3.11 also has no `RESUME`
# for the coordinate below to name.
#
# CPython-suite gap: `test_sys_settrace` never reads `f_lasti`, and the
# `test_frame` cases that do read it only on a frame that is already running.
#
# parity-tests reason: `f_lasti` is `_PyInterpreterFrame_LASTI * 2`
# (`PyUnstable_InterpreterFrame_GetLasti`, `Python/frame.c`), and the getter
# turns only a *negative* result into `-1` (`frame_lasti_get_impl`,
# `Objects/frameobject.c`).  A frame reached by the `call` event has not run a
# traceable instruction, but its instruction pointer is not a sentinel: the
# compiler-inserted prologue is consumed by frame setup, so the pointer sits on
# the first `RESUME` -- `code->_co_firsttraceable`.
#
# The gap is only visible when a prologue exists, because `_co_firsttraceable`
# is 0 without one.  All three shapes below are checked for that reason: a
# plain function has no prologue, a closure carries `COPY_FREE_VARS`, and a
# generator carries `RETURN_GENERATOR` / `POP_TOP`.
#
# A generator reached through `gi_frame` before its first `send` is the one
# frame that has run an instruction without reaching its `RESUME`:
# `RETURN_GENERATOR` is what produced the object being read, so the pointer
# rests on the instruction after it.  That coordinate is not a constant either
# -- a closure generator's `COPY_FREE_VARS` shifts `RETURN_GENERATOR` along --
# and it is one unit short of the `call` event's, so a runtime answering both
# from `_co_firsttraceable` is wrong for exactly one of them.

import dis
import sys


def first_resume_offset(code):
    """The byte offset of `code`'s first `RESUME` -- `_co_firsttraceable * 2`."""
    for instruction in dis.get_instructions(code):
        if instruction.opname == "RESUME":
            return instruction.offset
    raise AssertionError(f"no RESUME in {code.co_name}")


def after_return_generator_offset(code):
    """The byte offset of the instruction following `code`'s `RETURN_GENERATOR`."""
    instructions = list(dis.get_instructions(code))
    for position, instruction in enumerate(instructions):
        if instruction.opname == "RETURN_GENERATOR":
            return instructions[position + 1].offset
    raise AssertionError(f"no RETURN_GENERATOR in {code.co_name}")


def first_call_event_lasti(call_it, name):
    """`f_lasti` as the *first* `call` event for `name`'s frame reports it.

    A generator gets one `call` event per resumption, so only the first names
    the frame that has yet to run a traceable instruction.
    """
    seen = []

    def tracer(frame, event, arg):
        if frame.f_code.co_name == name and event == "call":
            seen.append(frame.f_lasti)
        return tracer

    sys.settrace(tracer)
    try:
        call_it()
    finally:
        sys.settrace(None)
    assert seen, name
    return seen[0]


def a_plain_function_resumes_at_offset_zero():
    def plain(n):
        return n + 1

    assert first_resume_offset(plain.__code__) == 0, "expected no prologue"
    assert first_call_event_lasti(lambda: plain(1), "plain") == 0


def a_closure_resumes_past_its_copy_free_vars():
    cell = 7

    def reads_the_cell(n):
        return cell + n

    prologue = [i.opname for i in dis.get_instructions(reads_the_cell.__code__)][:1]
    assert prologue == ["COPY_FREE_VARS"], prologue

    resume = first_resume_offset(reads_the_cell.__code__)
    assert resume > 0, resume
    assert first_call_event_lasti(lambda: reads_the_cell(1), "reads_the_cell") == resume


def a_generator_resumes_past_its_return_generator():
    def counts():
        yield 1

    prologue = [i.opname for i in dis.get_instructions(counts.__code__)][:2]
    assert prologue == ["RETURN_GENERATOR", "POP_TOP"], prologue

    resume = first_resume_offset(counts.__code__)
    assert resume > 0, resume
    assert first_call_event_lasti(lambda: list(counts()), "counts") == resume


def an_unstarted_generator_rests_after_its_return_generator():
    def counts():
        yield 1

    cell = 7

    def counts_from_a_cell():
        yield cell

    for maker in (counts, counts_from_a_cell):
        code = maker.__code__
        resting = after_return_generator_offset(code)
        # One unit short of where the `call` event reports the same frame.
        assert resting == first_resume_offset(code) - 2, (code.co_name, resting)
        # The generator is held for the read: a frame object outliving its
        # generator takes ownership of the frame and reports a coordinate past
        # the `RESUME` instead, which is not what this pins.
        held = maker()
        assert held.gi_frame.f_lasti == resting, code.co_name

    # `COPY_FREE_VARS` shifts the coordinate, so it is not a constant.
    plain = after_return_generator_offset(counts.__code__)
    closure = after_return_generator_offset(counts_from_a_cell.__code__)
    assert closure > plain, (plain, closure)


a_plain_function_resumes_at_offset_zero()
a_closure_resumes_past_its_copy_free_vars()
a_generator_resumes_past_its_return_generator()
an_unstarted_generator_rests_after_its_return_generator()

print("OK")
