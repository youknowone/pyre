# pyre-check: platforms=win32
# pyre-check: pypy-diverges: W_OSError.descr_str treats a None winerror as absent (`if self.w_winerror`), so PyPy keeps the errno form.
# CPython-suite gap: test_exceptions stringifies an integer winerror or no winerror at all. A None fourth argument is not asserted.
# parity-tests reason: OSError_str prefers a non-NULL winerror over errno, and None is a live object, so both filenames are still reported under [WinError None].
def a_none_winerror_still_names_both_filenames():
    exc = OSError(2, 'nope', 'a.txt', None, 'b.txt')
    assert exc.filename == 'a.txt', exc.filename
    assert exc.filename2 == 'b.txt', exc.filename2
    assert str(exc) == "[WinError None] nope: 'a.txt' -> 'b.txt'", str(exc)


a_none_winerror_still_names_both_filenames()
print('OK')
