# CPython-suite gap: displayhook tests do not collect while a sink replaces write.
# parity-tests reason: this guards pyre/PyPy moving-GC roots across callbacks.

# `sys.displayhook` writes twice, and either write may unbind the sink.
#
# The hook renders `repr(obj)` and then writes the repr and a newline.  Both
# the repr and the write re-enter the eval loop, so either can rebind
# `sys.stdout` and drop the last reference to the object the hook is writing
# through — after which reusing that object reads reclaimed memory.
# `pypy/module/sys/app.py:252-253` fetches `sys_stdout()` once per write.
#
# Which sink receives the newline is deliberately not asserted: CPython keeps
# the sink it resolved first and writes the newline to that one (gh-130163),
# while PyPy re-fetches and writes to whatever `sys.stdout` names by then.
# What both guarantee is that the two writes happen and that neither goes
# through a reclaimed object, so that is what this checks.
import gc
import sys


def check_write_rebinds_stdout():
    """The repr write drops the sink, so the newline write cannot reuse it."""
    log = []

    def make_sink(tag):
        def write(s):
            log.append((tag, s))
            if tag == "first":
                sys.stdout = make_sink("second")  # `first` loses its only reference
                gc.collect()
            return len(s)

        class Sink:
            pass

        sink = Sink()
        # A plain function in the instance dict: the call passes no `self`, so
        # while `write` runs, nothing but `sys.stdout` keeps the sink alive.  A
        # bound method would root it as the callee's argument and the hazard
        # would never arise.
        sink.write = write
        sink.flush = lambda: None
        return sink

    saved = sys.stdout
    try:
        sys.stdout = make_sink("first")  # the only reference
        sys.displayhook(42)
    finally:
        sys.stdout = saved
    # Non-vacuous: both writes really ran, and the repr went to the sink that
    # was current when the hook started.
    assert [s for _tag, s in log] == ["42", "\n"], log
    assert log[0][0] == "first", log


check_write_rebinds_stdout()

print("OK")
