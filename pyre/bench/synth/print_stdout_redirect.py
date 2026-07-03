# bltinmodule.c builtin_print / app_io.py print_ resolve a `file is None`
# default to the live `sys.stdout` each call, so rebinding `sys.stdout` from
# Python redirects `print()`. A `None` sys.stdout emits nothing.
import io
import sys


def main():
    # Default: prints to the real stdout.
    print('before')

    # Rebinding sys.stdout redirects print() into the new sink.
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    print('captured', 1, 2, sep='-', end='!\n')
    sys.stdout = saved
    print('captured =', repr(buf.getvalue()))

    # A None sys.stdout drops the output silently, then restores.
    sys.stdout = None
    print('dropped while stdout is None')
    sys.stdout = saved
    print('after None')

    # flush=True routes through the sink's flush() too.
    buf2 = io.StringIO()
    sys.stdout = buf2
    print('flushed', flush=True)
    sys.stdout = saved
    print('flushed =', repr(buf2.getvalue()))


main()
