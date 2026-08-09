# `print()` keeps using the sink it resolved, after running Python that can
# unbind it.
#
# With no `file=`, `print` resolves `sys.stdout` once and then calls `str()` on
# each argument and `write` on the sink.  Both re-enter the eval loop, so a
# `__str__` that rebinds `sys.stdout` drops the last reference to the object
# `print` is holding, and the collection that follows reclaims it — the write
# after it goes through a freed object.  The sink therefore has to be held for
# the whole call, not read once into a local.
#
# `check_str_rebinds_stdout` is CPython's `test_print.test_gh130163`.  The other
# two move the rebinding into the sink's own `write`: `write` is a plain
# function in the instance dict, so the call does not pass the sink as `self`
# and nothing but `print` itself keeps it alive while it runs.  That covers the
# separator and terminator writes, which take a different path from the
# argument write.
import gc
import sys
from io import StringIO


def check_str_rebinds_stdout():
    class X:
        def __str__(self):
            sys.stdout = StringIO()
            gc.collect()
            return "foo"

    saved = sys.stdout
    try:
        sys.stdout = StringIO()  # the only reference
        print(X())
    finally:
        sys.stdout = saved


def check_write_rebinds_stdout(*args, **kwargs):
    """`print(*args, **kwargs)` where the first write unbinds the sink."""
    written = []

    def writer(s):
        written.append(s)
        sys.stdout = StringIO()  # the running sink loses its only reference
        gc.collect()
        return len(s)

    def make_sink():
        class Sink:
            pass

        sink = Sink()
        sink.write = writer  # plain function: the call passes no `self`
        sink.flush = lambda: None
        return sink

    saved = sys.stdout
    try:
        sys.stdout = make_sink()  # the only reference
        print(*args, **kwargs)
    finally:
        sys.stdout = saved
    # Non-vacuous: the sink really was written through more than once, so the
    # writes after the rebinding are the ones the fix has to survive.
    assert len(written) > 1, written


check_str_rebinds_stdout()
# Two arguments so the separator write runs between them, and the default
# terminator write runs after.
check_write_rebinds_stdout("a", "b")
# The explicit `sep=` / `end=` objects take the argument path instead.
check_write_rebinds_stdout("a", "b", sep="-", end="!\n")
# `flush=True` calls the sink once more, after every write.
check_write_rebinds_stdout("a", "b", flush=True)

print("OK")
