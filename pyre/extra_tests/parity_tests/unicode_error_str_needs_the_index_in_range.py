# pyre-check: pypy-diverges: pins 3.14's range-form fallback when the offending
# index lies outside the object; pypy3 reads the object unguarded, so the same
# calls raise IndexError, and a negative start reads from the end instead.
#
# CPython-suite gap: `test_codeccallbacks` and `test_exceptions` build
# `UnicodeDecodeError` / `UnicodeEncodeError` / `UnicodeTranslateError` by hand
# and check the attributes and the ordinary one-character message, but every
# instance they stringify has its `start` inside the object.  Nothing there
# stringifies one whose `start` is past the end, or negative, so nothing notices
# a `__str__` that reports a character it could not read.
#
# parity-tests reason: all three `__str__` implementations take the
# single-character message only when the slice is genuinely inside the object --
# `start >= 0 and start < len and end >= 0 and end <= len and end == start + 1`
# -- and otherwise report the range form.  An implementation that keeps the
# single-character shape and substitutes a placeholder for the character it
# could not read prints a message no other implementation produces, and it does
# so only for hand-built instances, which no codec round trip reaches.
def a_decode_error_past_the_end_reports_a_range():
    exc = UnicodeDecodeError('utf-8', b'', 0, 1, 'bad')
    assert str(exc) == "'utf-8' codec can't decode bytes in position 0-0: bad", str(exc)
    exc = UnicodeDecodeError('utf-8', b'ab', 5, 6, 'bad')
    assert str(exc) == "'utf-8' codec can't decode bytes in position 5-5: bad", str(exc)


def a_negative_decode_start_reports_a_range():
    exc = UnicodeDecodeError('utf-8', b'ab', -1, 0, 'bad')
    assert str(exc) == "'utf-8' codec can't decode bytes in position -1--1: bad", str(exc)


def an_in_range_decode_byte_still_names_the_byte():
    exc = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'bad')
    assert str(exc) == "'utf-8' codec can't decode byte 0xff in position 0: bad", str(exc)
    exc = UnicodeDecodeError('utf-8', b'ab', 1, 2, 'bad')
    assert str(exc) == "'utf-8' codec can't decode byte 0x62 in position 1: bad", str(exc)


def an_encode_error_past_the_end_reports_a_range():
    exc = UnicodeEncodeError('utf-8', '', 0, 1, 'bad')
    assert str(exc) == "'utf-8' codec can't encode characters in position 0-0: bad", str(exc)
    exc = UnicodeEncodeError('utf-8', 'ab', 5, 6, 'bad')
    assert str(exc) == "'utf-8' codec can't encode characters in position 5-5: bad", str(exc)
    exc = UnicodeEncodeError('utf-8', 'ab', -1, 0, 'bad')
    assert str(exc) == "'utf-8' codec can't encode characters in position -1--1: bad", str(exc)


def an_in_range_encode_character_still_names_the_character():
    exc = UnicodeEncodeError('utf-8', '\ud800', 0, 1, 'bad')
    assert str(exc) == "'utf-8' codec can't encode character '\\ud800' in position 0: bad", str(exc)


def a_translate_error_past_the_end_reports_a_range():
    exc = UnicodeTranslateError('', 0, 1, 'bad')
    assert str(exc) == "can't translate characters in position 0-0: bad", str(exc)
    exc = UnicodeTranslateError('ab', 5, 6, 'bad')
    assert str(exc) == "can't translate characters in position 5-5: bad", str(exc)


def an_in_range_translate_character_still_names_the_character():
    exc = UnicodeTranslateError('a', 0, 1, 'bad')
    assert str(exc) == "can't translate character '\\x61' in position 0: bad", str(exc)


a_decode_error_past_the_end_reports_a_range()
a_negative_decode_start_reports_a_range()
an_in_range_decode_byte_still_names_the_byte()
an_encode_error_past_the_end_reports_a_range()
an_in_range_encode_character_still_names_the_character()
a_translate_error_past_the_end_reports_a_range()
an_in_range_translate_character_still_names_the_character()
print('OK')
