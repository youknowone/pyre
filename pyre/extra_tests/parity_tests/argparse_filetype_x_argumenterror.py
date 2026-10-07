# CPython-suite gap: test_argparse TestFileTypeX.test_failures_one_group_listargs
# ERROR'd with argparse.ArgumentError: argument spam: can't open 'writable'
# after FileType('x') converted FileExistsError through ArgumentTypeError.
# parity-tests reason: `_parse_known_args2` catches ArgumentError and calls
# error() (SystemExit). After the no-error path of parse_args is compiled,
# that except must still match; a folded CHECK_EXC_MATCH or an exception
# bridge that skips the handler leaks ArgumentError as a unittest ERROR.

import argparse
import io
import os
import sys
import tempfile
import warnings

WARMUP = 4000
SWITCHED = 400


def make_parser():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        parser = argparse.ArgumentParser()
        # test_failures_one_group_listargs adds both arguments under one group.
        group = parser.add_argument_group("foo")
        group.add_argument("-x", type=argparse.FileType("x"))
        group.add_argument("spam", type=argparse.FileType("x"))
    return parser


def parse_args_quiet(parser, args):
    # ErrorRaisingArgumentParser.parse_args redirects stderr the same way:
    # error() writes usage then SystemExit; ArgumentError must not leak.
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        return parser.parse_args(args)
    finally:
        sys.stderr = old_stderr


def parse_existing_exclusive(parser, name):
    try:
        parse_args_quiet(parser, [name])
    except SystemExit:
        return "exit"
    except argparse.ArgumentError:
        return "leak"
    return "ok"


def main():
    tmp = tempfile.mkdtemp()
    old = os.getcwd()
    os.chdir(tmp)
    try:
        with open("writable", "w", encoding="utf-8") as file:
            file.write("writable")
        parser = make_parser()

        # Compile the no-exception path first, as the rest of test_argparse
        # does before TestFileTypeX. FileType('x') creates the file, so each
        # success uses a fresh name and closes the handles.
        for i in range(WARMUP):
            a = "ok_a_%d" % i
            b = "ok_b_%d" % i
            ns = parse_args_quiet(parser, ["-x", a, b])
            ns.x.close()
            ns.spam.close()
            os.remove(a)
            os.remove(b)

        counts = {"exit": 0, "leak": 0, "ok": 0}
        for _ in range(SWITCHED):
            counts[parse_existing_exclusive(parser, "writable")] += 1
        assert counts["leak"] == 0, counts
        assert counts["ok"] == 0, counts
        assert counts["exit"] == SWITCHED, counts
    finally:
        os.chdir(old)


main()
print("OK")
