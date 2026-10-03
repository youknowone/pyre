# CPython-suite gap: `test_exceptions` and `test_os` construct `OSError` with
# two and with three arguments, but the three-argument assertions they make read
# `errno` / `strerror` / `filename` back as attributes; none of them stringifies
# a `BlockingIOError` whose third argument is `characters_written` rather than a
# filename, so nothing there notices a `__str__` that reads `args[2]`.
#
# parity-tests reason: `OSError.__str__` reports the `filename` and `filename2`
# slots, not positional arguments.  `OSError.__init__` only fills those slots
# for the 2-to-5 argument forms that have a filename, and for the
# `characters_written` subclasses the third argument goes to that slot instead,
# leaving `filename` unset.  A `__str__` that falls back to `args[2]` appends a
# filename that the exception does not have, and the difference shows only in
# the message -- every attribute read still answers correctly.
#
# CPython 3.14 and PyPy agree on every arm below.  A None `winerror`
# is different on Windows: `OSError_str` prints it, and that spelling
# lives in `oserror_winerror_none_names_both_filenames.py`.
import errno
import sys


def a_characters_written_argument_is_not_a_filename():
    exc = BlockingIOError(11, 'again', 4)
    assert exc.characters_written == 4, exc.characters_written
    assert exc.filename is None, exc.filename
    assert str(exc) == '[Errno 11] again', str(exc)


def a_third_argument_is_a_filename_for_a_plain_oserror():
    exc = OSError(11, 'again', 'f.txt')
    assert exc.filename == 'f.txt', exc.filename
    assert str(exc) == "[Errno 11] again: 'f.txt'", str(exc)


def a_fifth_argument_is_the_second_filename():
    exc = OSError(2, 'nope', 'a.txt', None, 'b.txt')
    assert exc.filename == 'a.txt', exc.filename
    assert exc.filename2 == 'b.txt', exc.filename2
    # Windows keeps the None winerror slot, so `OSError_str` takes the
    # WinError arm.  `W_OSError.descr_str` treats that None as absent.
    if sys.platform == "win32":
        return
    assert str(exc) == "[Errno 2] nope: 'a.txt' -> 'b.txt'", str(exc)


def the_two_argument_form_reports_no_filename():
    assert str(OSError(2, 'nope')) == '[Errno 2] nope', str(OSError(2, 'nope'))
    assert str(BlockingIOError(11, 'again')) == '[Errno 11] again'


def an_errno_that_selects_a_subclass_still_reports_the_pair():
    exc = OSError(errno.ENOENT, 'nope')
    assert type(exc) is FileNotFoundError, type(exc)
    assert str(exc) == '[Errno 2] nope', str(exc)


def a_stored_none_filename_is_part_of_the_message():
    exc = OSError(2, 'nope', 'a.txt')
    exc.filename = None
    assert str(exc) == '[Errno 2] nope: None', str(exc)
    assert exc.__reduce__()[1] == (2, 'nope', None), exc.__reduce__()
    if sys.platform == "win32":
        return
    exc = OSError(2, 'nope', 'a.txt', None, 'b.txt')
    exc.filename2 = None
    assert str(exc) == "[Errno 2] nope: 'a.txt' -> None", str(exc)
    assert exc.__reduce__()[1] == (2, 'nope', 'a.txt', None, None), exc.__reduce__()


def a_constructor_none_does_not_fill_the_filename_slot():
    exc = OSError(2, 'nope', None)
    assert exc.args == (2, 'nope', None), exc.args
    assert exc.filename is None
    assert str(exc) == '[Errno 2] nope', str(exc)
    assert exc.__reduce__()[1] == (2, 'nope', None), exc.__reduce__()


a_characters_written_argument_is_not_a_filename()
a_third_argument_is_a_filename_for_a_plain_oserror()
a_fifth_argument_is_the_second_filename()
the_two_argument_form_reports_no_filename()
an_errno_that_selects_a_subclass_still_reports_the_pair()
a_stored_none_filename_is_part_of_the_message()
a_constructor_none_does_not_fill_the_filename_slot()
print('OK')
