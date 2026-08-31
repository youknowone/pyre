# CPython-suite gap: print and displayhook tests do not collect while a write
# callback replaces the active stdout sink.
# parity-tests reason: this guards pyre/PyPy moving-GC roots across the two
# stdout callback paths.

"""The stdout sink remains live across displayhook and print callbacks."""

import gc
import sys
from io import StringIO


def check_displayhook_sink():
    log = []

    def make_sink(tag):
        def write(text):
            log.append((tag, text))
            if tag == "first":
                sys.stdout = make_sink("second")
                gc.collect()
            return len(text)

        class Sink:
            pass

        sink = Sink()
        sink.write = write
        sink.flush = lambda: None
        return sink

    saved = sys.stdout
    try:
        sys.stdout = make_sink("first")
        sys.displayhook(42)
    finally:
        sys.stdout = saved
    assert [text for _tag, text in log] == ["42", "\n"]
    assert log[0][0] == "first"


def check_print_sink(*args, **kwargs):
    written = []

    def write(text):
        written.append(text)
        sys.stdout = StringIO()
        gc.collect()
        return len(text)

    class Sink:
        pass

    sink = Sink()
    sink.write = write
    sink.flush = lambda: None
    saved = sys.stdout
    try:
        sys.stdout = sink
        del sink
        print(*args, **kwargs)
    finally:
        sys.stdout = saved
    assert len(written) > 1


check_displayhook_sink()
check_print_sink("a", "b")
check_print_sink("a", "b", sep="-", end="!\n")
check_print_sink("a", "b", flush=True)

print("OK")
