# pyre-check: pypy-diverges: pins that a StopIteration caught by FOR_ITER's exhaustion arm reports no `exception` event; pypy3 reports ('consume_generator', 'StopIteration')
# CPython-suite gap: `test_sys_settrace` traces loops and it traces raising
# code, but never a loop whose iterator ends by raising StopIteration from a
# Python-level `__next__` while a tracer is installed.  A runtime that
# swallowed that StopIteration without reporting it would pass the whole
# module.
#
# parity-tests reason: FOR_ITER's exhaustion arm catches the StopIteration and
# jumps, so the consuming frame never dispatches it and never reaches the
# code that reports an exception to the tracer.  Whether an `exception` event
# is delivered there is therefore a decision the exhaustion arm has to make on
# its own, and it is one a debugger sees: `pdb` stops on `exception` events.
#
# The rule is not "always" and not "never".  3.14 reports the event exactly
# when the StopIteration carries a traceback -- which is to say, when it was
# raised by Python code rather than signalled by an iterator implemented
# natively.  A generator ending, a list ending and a range ending all deliver
# nothing; a class whose `__next__` raises delivers one.  That asymmetry is
# what this pins: reporting always would fire on every `for` over a list, and
# reporting never would lose the one case a debugger cares about.
#
# PyPy 7.3.20 fails `a_natively_signalled_end_reports_nothing` on the generator
# case: it reports every generator-iterator unconditionally, which its own
# comment calls "an approximative rule" for a case it says it cannot emulate.
# The rule pinned here is 3.14's, so this file is a place the two references
# disagree and CPython wins.
import sys


def exception_events_in(consumer):
    """The 'exception' events a tracer sees while `consumer` runs."""
    seen = []

    def record(frame, event, arg):
        if event == 'exception':
            seen.append((frame.f_code.co_name, arg[0].__name__))
        return record

    sys.settrace(record)
    try:
        consumer()
    finally:
        sys.settrace(None)
    return seen


def a_python_level_next_reports_to_the_consuming_frame():
    class Ending:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    def consume():
        for _ in Ending():
            raise AssertionError('the iterator was not empty')

    seen = exception_events_in(consume)
    # Two events: the raise inside `__next__`, and the report FOR_ITER makes
    # in the frame running the loop.  The second is the one under test.
    assert seen == [('__next__', 'StopIteration'), ('consume', 'StopIteration')], seen


def the_rule_is_the_traceback_and_not_the_iterator():
    """`map`'s `__next__` is native, but the StopIteration is not."""

    def stop(_):
        raise StopIteration

    def consume():
        for _ in map(stop, [1, 2]):
            raise AssertionError('the map was not empty')

    # The exhaustion came out of a native `__next__`, so a rule that keyed off
    # the iterator's kind would report nothing here.  It travelled through
    # `stop`'s frame on the way, which is what the consuming frame reports on.
    assert exception_events_in(consume) == [
        ('stop', 'StopIteration'),
        ('consume', 'StopIteration'),
    ], 'unexpected events'


def a_natively_signalled_end_reports_nothing():
    def consume_generator():
        def empty():
            return
            yield

        for _ in empty():
            raise AssertionError('the generator was not empty')

    def consume_list():
        for _ in [1, 2]:
            pass

    def consume_range():
        for _ in range(2):
            pass

    def consume_dict():
        for _ in {'a': 1}:
            pass

    def consume_callable_iter():
        box = [0]

        def step():
            box[0] += 1
            return box[0]

        # `iter(callable, sentinel)` ends by comparing, not by raising through
        # `step`, so this is a native end even though `step` is Python.
        for _ in iter(step, 3):
            pass

    for consumer in (
        consume_generator,
        consume_list,
        consume_range,
        consume_dict,
        consume_callable_iter,
    ):
        assert exception_events_in(consumer) == [], consumer.__name__


def the_loop_still_ends_normally():
    class Ending:
        def __init__(self, n):
            self.n = n

        def __iter__(self):
            return self

        def __next__(self):
            if self.n == 0:
                raise StopIteration
            self.n -= 1
            return self.n

    collected = []

    def consume():
        for item in Ending(3):
            collected.append(item)

    # The report must not disturb what the loop collects, and must not leave
    # the StopIteration visible to the loop body.
    assert exception_events_in(consume) == [
        ('__next__', 'StopIteration'),
        ('consume', 'StopIteration'),
    ], 'unexpected events'
    assert collected == [2, 1, 0], collected


a_python_level_next_reports_to_the_consuming_frame()
the_rule_is_the_traceback_and_not_the_iterator()
a_natively_signalled_end_reports_nothing()
the_loop_still_ends_normally()
print('OK')
